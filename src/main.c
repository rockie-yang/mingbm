/*
 * MinGBM - Minimal Gradient Boosting Machine
 * Usage: ./mingbm <csv_file> [n_trees] [learning_rate]
 */

#include "../header.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    const char *csv_file = argv[1];
    uint32_t n_trees = argc > 2 ? atoi(argv[2]) : 50;
    float learning_rate = argc > 3 ? atof(argv[3]) : 0.1f;

    print_color(CYAN, "=== MinGBM Training ===\n\n");

    // Detect and load dataset
    print_color(YELLOW, "[1/6] ");
    printf("Detecting CSV format...\n");
    uint32_t n_samples, n_features;
    if (detect_csv_format(csv_file, &n_samples, &n_features) < 0) {
        return 1;
    }
    print_color(GREEN, "  Detected: %u samples, %u features\n\n", n_samples, n_features);

    print_color(YELLOW, "[2/6] ");
    printf("Creating dataset mmap...\n");
    size_t dataset_size = calculate_dataset_size(n_samples, n_features);
    void *dataset = create_dataset_mmap("dataset.bin", n_samples, n_features);
    if (!dataset) return 1;

    if (load_csv_to_mmap(csv_file, dataset) < 0) {
        close_dataset_mmap(dataset, dataset_size);
        return 1;
    }
    print_color(GREEN, "  Dataset loaded (%.2f KB)\n\n", dataset_size / 1024.0);

    // Create model
    print_color(YELLOW, "[3/6] ");
    printf("Creating model structure...\n");
    uint32_t max_nodes = (1 << (MAX_DEPTH + 1)) - 1;
    size_t model_size = calculate_model_size(n_trees, max_nodes);
    void *model = create_model_mmap("model.bin", n_trees, max_nodes);
    if (!model) {
        close_dataset_mmap(dataset, dataset_size);
        return 1;
    }
    print_color(GREEN, "  Model: %u trees, max %u nodes/tree (%.2f KB)\n\n",
           n_trees, max_nodes, model_size / 1024.0);

    // Create binned dataset
    print_color(YELLOW, "[4/6] ");
    printf("Creating binned dataset...\n");
    size_t binned_size = calculate_binned_size(n_samples, n_features);
    void *binned_dataset = create_binned_mmap("binned.bin", n_samples, n_features, MAX_BINS);
    if (!binned_dataset) {
        close_model_mmap(model, model_size);
        close_dataset_mmap(dataset, dataset_size);
        return 1;
    }

    // Initialize training context
    TrainContext ctx = {0};
    ctx.dataset = dataset;
    ctx.binned_dataset = binned_dataset;
    ctx.model = model;
    ctx.n_samples = n_samples;
    ctx.n_features = n_features;
    ctx.gradients = (float*)malloc(n_samples * sizeof(float));
    ctx.hessians = (float*)malloc(n_samples * sizeof(float));
    ctx.predictions = (float*)malloc(n_samples * sizeof(float));

    create_bin_mappers(&ctx);
    print_color(GREEN, "  Binned dataset (%.2f KB)\n\n", binned_size / 1024.0);

    // Initialize EFB (Exclusive Feature Bundling)
    init_efb(&ctx);

    // Train model
    print_color(YELLOW, "[5/6] ");
    printf("Training GBM...\n");
    train_gbm(&ctx, n_trees, learning_rate);

    // Evaluate
    printf("\n");
    print_color(YELLOW, "[6/6] ");
    printf("Evaluating model...\n");
    evaluate_model(&ctx);

    // Cleanup
    free_efb(&ctx);
    free(ctx.gradients);
    free(ctx.hessians);
    free(ctx.predictions);
    free(ctx.bin_mappers);
    close_binned_mmap(binned_dataset, binned_size);
    close_model_mmap(model, model_size);
    close_dataset_mmap(dataset, dataset_size);

    printf("\n");
    print_color(CYAN, "=== Training Complete ===\n");

    return 0;
}
