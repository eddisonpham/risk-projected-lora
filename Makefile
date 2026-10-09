# Thin shim. `make` is unavailable on some dev boxes (Windows w/o MSYS), so the
# real logic lives in tasks.py; both entry points stay equivalent.
#   make setup | test | test-sim | test-gpu | lint | smoke | report
#   make exp EXP=configs/exp/<name>.yaml
PY ?= python
export PYTHONPATH := $(CURDIR)/src

.PHONY: help setup bootstrap-sim env-lock test test-sim test-gpu test-all lint typecheck fmt smoke exp report clean

help:
	@$(PY) tasks.py --help

setup:            ; @$(PY) tasks.py setup
bootstrap-sim:    ; @$(PY) tasks.py bootstrap-sim
env-lock:         ; @$(PY) tasks.py env-lock
test:             ; @$(PY) tasks.py test
test-sim:         ; @$(PY) tasks.py test-sim
test-gpu:         ; @$(PY) tasks.py test-gpu
test-all:         ; @$(PY) tasks.py test-all
lint:             ; @$(PY) tasks.py lint
typecheck:        ; @$(PY) tasks.py typecheck
fmt:              ; @$(PY) tasks.py fmt
smoke:            ; @$(PY) tasks.py smoke
exp:              ; @$(PY) tasks.py exp EXP=$(EXP)
report:           ; @$(PY) tasks.py report RUN=$(RUN)
clean:            ; @$(PY) tasks.py clean
