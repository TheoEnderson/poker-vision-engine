"""Benchmark 4E-B3, somente stdout; não usa OCR, arquivos ou aleatoriedade.

Executar: venv/bin/python -B -m scripts.benchmark_cfr
Cada amostra cria um solver novo. Árvores são construídas antes da medição.
NashConv é soma dos ganhos de melhor resposta (exploitability = NashConv/2
na convenção zero-sum de dois jogadores). L1 contra 1000 é só estabilidade.
"""

import argparse
import gc
import json
import math
import platform
import statistics
import time

from tests.cfr_scenarios import (HERO, ITERATIONS, count_nodes,
                                 full_information_gap, matrix_metrics, scenarios)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", help="Subconjunto opcional dos cenários.")
    args = parser.parse_args()
    if args.repetitions < 2 or args.batches < 2:
        parser.error("Use ao menos duas repetições e dois lotes independentes.")
    start = time.perf_counter()
    cases = scenarios()
    if args.scenarios:
        if any(name not in cases for name in args.scenarios):
            parser.error("Cenário desconhecido.")
        cases = {name: cases[name] for name in args.scenarios}
    emit(type="environment", python=platform.python_version(), platform=platform.platform(),
         repetitions=args.repetitions, batches=args.batches, gc_enabled=gc.isenabled(),
         construction_ms=(time.perf_counter()-start)*1000,
         timing_scope="novo solver + solve; sem construção da árvore/OCR/Monte Carlo")

    for name, (root, factory, equity) in cases.items():
        reference = factory().solve(root, HERO, equity, 1000)
        previous = None
        for iterations in ITERATIONS:
            solver = factory()
            strategy = solver.solve(root, HERO, equity, iterations)
            gap = (matrix_metrics(solver, root)["nash_conv"] if name == "matrix_mixed"
                   else full_information_gap(root, solver, equity))
            extra = matrix_metrics(solver, root) if name == "matrix_mixed" else {}
            emit(type="convergence", scenario=name, iterations=iterations,
                 tree_nodes=count_nodes(root), strategy=strategy, nash_conv=gap,
                 l1_to_1000=math.fsum(abs(strategy[a]-reference[a]) for a in strategy),
                 l1_to_previous=None if previous is None else
                     math.fsum(abs(strategy[a]-previous[a]) for a in strategy), **{
                         key: value for key, value in extra.items() if key != "nash_conv"})
            previous = strategy

    configurations = [(name, n) for name in cases for n in ITERATIONS]
    for batch in range(args.batches):
        # Mesmo aquecimento em cada lote, sem reutilizar tabelas de regrets.
        for root, factory, equity in cases.values():
            factory().solve(root, HERO, equity, 30)
        samples = {key: [] for key in configurations}
        for repetition in range(args.repetitions):
            offset = (repetition * 11 + batch * 7) % len(configurations)
            order = configurations[offset:] + configurations[:offset]
            for name, iterations in order:
                root, factory, equity = cases[name]
                start_ns = time.perf_counter_ns()
                factory().solve(root, HERO, equity, iterations)
                samples[(name, iterations)].append((time.perf_counter_ns()-start_ns)/1e6)
        for (name, iterations), values in samples.items():
            nodes = count_nodes(cases[name][0])
            emit(type="benchmark", batch=batch+1, scenario=name, iterations=iterations,
                 tree_nodes=nodes, node_visits=nodes*iterations, samples_ms=values,
                 mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                 p95_ms=percentile(values, .95), stdev_ms=statistics.stdev(values),
                 min_ms=min(values), max_ms=max(values))


if __name__ == "__main__":
    main()
