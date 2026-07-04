# latticecp -- one target per stage; SLURM-compatible where it matters.
PY := python3
OLD ?=            # path to the audited results_june.csv for `make diff-old`

.PHONY: install test test-full quick scan verify-artifacts bench-slurm \
        bench-local merge analyze figures audit diff-old clean

install:
	pip install -e .

test:                       ## fast: theorem tests + pipeline plumbing (~2 min)
	$(PY) tests/test_theorems.py
	$(PY) tests/test_pipeline.py
	$(PY) tests/test_driver_rows.py

test-full:                  ## adds the Monte-Carlo validity tests (~10 min)
	$(PY) tests/test_theorems.py
	$(PY) tests/test_pipeline.py --full
	$(PY) tests/test_driver_rows.py

quick:                      ## tiny local benchmark end-to-end (needs data cache or network)
	$(PY) stages/s03_run_benchmark.py --local --quick
	$(PY) stages/s04_merge.py
	$(PY) stages/s05_analyze.py results/results_bench.csv | head -60

verify-artifacts:           ## determinism check: frozen scan -> shortlist must match frozen shortlist
	$(PY) stages/s02_make_shortlist.py --scan artifacts/master_scan.csv --check artifacts/shortlist.csv

scan:                       ## EXPENSIVE full view scan from the frozen catalog (optional regen)
	$(PY) stages/s01_scan_views.py --workers 8

bench-slurm:                ## the full benchmark as a SLURM array
	mkdir -p logs
	sbatch slurm/submit_benchmark.sh

bench-local:                ## the full benchmark on one machine (days; prefer bench-slurm)
	$(PY) stages/s03_run_benchmark.py --local --workers 8

merge:
	$(PY) stages/s04_merge.py

analyze:
	$(PY) stages/s05_analyze.py results/results_bench.csv

figures:
	$(PY) stages/s06_figures.py results/results_bench.csv

audit:
	$(PY) stages/s07_audit.py results/results_bench.csv

diff-old:                   ## acceptance test: make diff-old OLD=/path/to/results_june.csv
	$(PY) scripts/diff_reproduction.py $(OLD) results/results_bench.csv

clean:
	rm -rf results_bench logs/*.out logs/*.err
