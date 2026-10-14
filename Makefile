# CoMPASS — reproduction entry points.
#
#   make setup       create .venv and install pinned analysis dependencies
#   make reproduce   regenerate every number and figure in the paper
#   make verify      assert the headline values match what the paper reports
#   make clean       remove regenerated analysis output
#
# `reproduce` needs no API keys: it re-analyses the released judge scores in
# full_matrix_v6/. Regenerating the transcripts themselves does need keys --
# see README "Full pipeline".

VENV    := .venv
PY      := $(VENV)/bin/python
MATRIX  := full_matrix_v6
JUDGE_A := $(MATRIX)/results/judge_a_scores
JUDGE_B := $(MATRIX)/results/judge_b_scores
TRANSCR := $(MATRIX)/transcripts
OUT_A   := analysis/judge_a
OUT_B   := analysis/judge_b

.PHONY: all setup reproduce verify clean

all: reproduce

$(VENV):
	python3 -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

setup: $(VENV)

# Primary judge (A) produces every headline number; judge B is the
# sensitivity analysis reported in the paper's judge-sensitivity section.
reproduce: setup
	@mkdir -p $(OUT_A) $(OUT_B)
	PYTHONPATH=. $(PY) scripts/analyze.py \
	  --results-dir $(JUDGE_A) --transcripts-dir $(TRANSCR) --output-dir $(OUT_A)
	PYTHONPATH=. $(PY) scripts/analyze.py \
	  --results-dir $(JUDGE_B) --transcripts-dir $(TRANSCR) --output-dir $(OUT_B)
	@echo
	@echo "Regenerated. Primary results: $(OUT_A)/  Sensitivity: $(OUT_B)/"
	@echo "Run 'make verify' to check them against the published values."

verify: reproduce
	@PYTHONPATH=. $(PY) scripts/verify_paper_numbers.py

clean:
	rm -rf analysis/
