# MinGBM - Minimal Gradient Boosting Machine

CC = clang
CFLAGS = -std=c11 -Wall -Wextra -O2 -march=native
LDFLAGS = -lm

SRC_DIR = src
TMP_DIR = tmp
BUILD_DIR = build

SRC_FILES = $(SRC_DIR)/dataset.c $(SRC_DIR)/model.c $(SRC_DIR)/train.c $(SRC_DIR)/utility.c
MAIN_FILES = $(SRC_DIR)/main.c $(SRC_FILES)
MAIN_BIN = $(BUILD_DIR)/mingbm

all: main

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TMP_DIR):
	@mkdir -p $(TMP_DIR)

main: $(BUILD_DIR) $(TMP_DIR) $(MAIN_FILES) header.h
	@echo "Building mingbm..."
	$(CC) $(CFLAGS) -o $(MAIN_BIN) $(MAIN_FILES) $(LDFLAGS)
	@echo "Built: $(MAIN_BIN)"

run: main
	@echo "Running MinGBM..."
	cd $(TMP_DIR) && ../$(MAIN_BIN) ../house-prices-advanced-regression-techniques/train.csv 50 0.1

compare: main
	@echo "\033[1;36m"
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║           MinGBM vs LightGBM Comparison                    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo "\033[0m"
	@echo ""
	@echo "\033[1;33m[1/2] Training MinGBM...\033[0m"
	cd $(TMP_DIR) && ../$(MAIN_BIN) ../house-prices-advanced-regression-techniques/train.csv 50 0.1
	@echo ""
	@echo "\033[1;33m[2/2] Training LightGBM...\033[0m"
	.venv/bin/python3 compare_lgbm.py house-prices-advanced-regression-techniques/train.csv 50 0.1
	@echo ""
	@echo "\033[1;36m════════════════════════════════════════════════════════════\033[0m"

clean:
	@echo "Cleaning..."
	rm -rf $(BUILD_DIR) $(TMP_DIR)
	@echo "Cleaned"

.PHONY: all main run compare clean
