PY ?= .venv/bin/python
RUN = PYTHONPATH=src $(PY) -m

.PHONY: setup data clean_data db analysis robustness plots test run_all clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# Downloads NBK (day-by-day, ~8k requests), FRED Brent and ECB reference rates into data/raw/.
# If a source is unreachable, place the file manually in data/raw/ (see README).
data:
	$(RUN) tenge_fx.download

clean_data:
	$(RUN) tenge_fx.clean

db:
	$(RUN) tenge_fx.build_db

analysis:
	$(RUN) tenge_fx.analysis

robustness:
	$(RUN) tenge_fx.robustness

plots:
	$(RUN) tenge_fx.plots

test:
	PYTHONPATH=src $(PY) -m pytest -q

# Full pipeline from data/raw/ (fixed seed 42).
run_all: clean_data db analysis robustness plots

clean:
	rm -rf data/processed/* reports/figures/*.png
