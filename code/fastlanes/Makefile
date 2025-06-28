VENV_DIR := .venv
PYTHON    := $(VENV_DIR)/bin/python
DEPS      := \
	pyfastlanes==0.1.3.post9 \
	pandas \
	duckdb \
	"gdown>=5.2.0" \
	packaging

all: run

$(VENV_DIR)/bin/activate:
	@echo "-- Creating virtual environment..."
	python3 -m venv $(VENV_DIR)

install: $(VENV_DIR)/bin/activate
	@echo "-- Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install $(DEPS)

run: install
	@echo "-- Running fcbench.py..."
	$(PYTHON) fcbench.py

clean:
	@echo "-- Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -name '__pycache__' -type d -exec rm -rf {} +
	rm -f data.fls
	# remove decoded.csv in these dirs
	@for dir in kv_cache_original diff_data; do \
	  rm -f $$dir/decoded.csv; \
	done
	# remove additional directories/files
	rm -rf HPC_TS_OBS HPC_TS_OBS_csvs DB_csvs DB

.PHONY: all install run clean
