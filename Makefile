# Makefile

# Python package files to create under each package
PY_INITS := \
	src/storage/__init__.py \
	src/etl/__init__.py \
	src/features/__init__.py \
	src/models/__init__.py \
	src/validation/__init__.py \
	src/inference/__init__.py \
	src/utils/__init__.py

# Other stub files
PY_FILES := \
	src/storage/db.py \
	src/storage/schemas.py \
	src/etl/extract.py \
	src/etl/transform.py \
	src/etl/load.py \
	src/features/temporal.py \
	src/features/spatial.py \
	src/features/interaction.py \
	src/models/baseline.py \
	src/models/stgcn.py \
	src/models/informer.py \
	src/models/train_eval.py \
	src/validation/cv_splitters.py \
	src/validation/metrics.py \
	src/validation/hyperopt.py \
	src/inference/predict.py \
	src/utils/config.py \
	src/utils/logger.py \
	src/utils/visualization.py

# Directories to create
DIRS := \
	data/raw \
	data/processed \
	data/external \
	docs \
	notebooks \
	src/storage \
	src/etl \
	src/features \
	src/models \
	src/validation \
	src/inference \
	src/utils \
	tests

.PHONY: init clean

init:
	mkdir -p $(DIRS)
	touch $(PY_INITS) $(PY_FILES)
	touch tests/test_extract.py tests/test_transform.py tests/test_baseline.py
	@echo "Project scaffold created."

clean:
	rm -rf data docs notebooks src tests
	@echo "All generated files and folders removed."