/*
 * MinGBM Utility Functions
 */

#include "../header.h"
#include <stdio.h>
#include <stdarg.h>

void print_color(const char *color, const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("%s", color);
    vprintf(format, args);
    printf("%s", RESET);
    va_end(args);
}

void print_help(const char *program_name) {
    print_color(CYAN, "\n=== MinGBM - Minimal Gradient Boosting Machine ===\n\n");

    printf("Usage:\n");
    print_color(YELLOW, "  %s", program_name);
    printf(" <csv_file> [n_trees] [learning_rate]\n\n");

    printf("Arguments:\n");
    print_color(GREEN, "  csv_file");
    printf("       Path to training CSV (required)\n");
    print_color(GREEN, "  n_trees");
    printf("        Number of trees (default: 50)\n");
    print_color(GREEN, "  learning_rate");
    printf("  Learning rate (default: 0.1)\n\n");

    printf("Example:\n");
    print_color(YELLOW, "  %s", program_name);
    printf(" data/train.csv 50 0.1\n\n");
}
