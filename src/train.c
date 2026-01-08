/*
 * MinGBM Training - Single-threaded histogram-based GBM
 */

#include "../header.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MIN_SAMPLES_LEAF 20
#define EFB_MIN_FEATURES 10  // Minimum features to enable EFB

// Quickselect partition for GOSS
static uint32_t partition_by_gradient(float *gradients, uint32_t *indices, uint32_t left, uint32_t right) {
    uint32_t pivot_idx = indices[right];
    float pivot_grad = fabsf(gradients[pivot_idx]);

    uint32_t i = left;
    for (uint32_t j = left; j < right; j++) {
        float curr_grad = fabsf(gradients[indices[j]]);
        if (curr_grad > pivot_grad) {
            uint32_t temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
            i++;
        }
    }

    uint32_t temp = indices[i];
    indices[i] = indices[right];
    indices[right] = temp;

    return i;
}

static int compare_floats(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static void quickselect(float *gradients, uint32_t *indices, uint32_t left, uint32_t right, uint32_t k) {
    if (left >= right || k == 0) return;

    uint32_t pivot_pos = partition_by_gradient(gradients, indices, left, right);

    if (pivot_pos == k) {
        return;
    } else if (pivot_pos > k) {
        quickselect(gradients, indices, left, pivot_pos - 1, k);
    } else {
        quickselect(gradients, indices, pivot_pos + 1, right, k);
    }
}

// GOSS sampling
uint32_t goss_sample_selection(TrainContext *ctx, uint32_t *samples, uint32_t n_samples) {
    float *sample_weights = ctx->sample_weights;
    uint32_t *sorted_indices = ctx->sorted_indices;
    float *gradients = ctx->gradients;

    if (!ctx->use_goss || n_samples < 100) {
        for (uint32_t i = 0; i < n_samples; i++) {
            sample_weights[samples[i]] = 1.0f;
        }
        return n_samples;
    }

    for (uint32_t i = 0; i < n_samples; i++) {
        sorted_indices[i] = samples[i];
    }

    uint32_t n_large = (uint32_t)(ctx->goss_alpha * n_samples);

    if (n_large > 0 && n_large < n_samples) {
        quickselect(gradients, sorted_indices, 0, n_samples - 1, n_large - 1);
    }
    uint32_t n_small_total = n_samples - n_large;
    uint32_t n_small_sample = (uint32_t)(ctx->goss_beta * n_small_total);

    uint32_t selected = 0;
    for (uint32_t i = 0; i < n_large; i++) {
        samples[selected] = sorted_indices[i];
        sample_weights[samples[selected]] = 1.0f;
        selected++;
    }

    float small_weight = n_small_total / (float)n_small_sample;
    for (uint32_t i = 0; i < n_small_sample; i++) {
        uint32_t rand_offset = (uint32_t)(rand() % (n_small_total - i));
        uint32_t selected_idx = n_large + i + rand_offset;

        uint32_t temp = sorted_indices[n_large + i];
        sorted_indices[n_large + i] = sorted_indices[selected_idx];
        sorted_indices[selected_idx] = temp;

        samples[selected] = sorted_indices[n_large + i];
        sample_weights[samples[selected]] = small_weight;
        selected++;
    }

    return selected;
}

// Create bin mappers with adaptive encoding
void create_bin_mappers(TrainContext *ctx) {
    ColumnDatasetHeader *dataset = (ColumnDatasetHeader*)ctx->dataset;
    BinnedDatasetHeader *binned = (BinnedDatasetHeader*)ctx->binned_dataset;
    FeatureMetadata *metadata = binned->metadata.ptr;
    ctx->bin_mappers = (BinMapper*)malloc(ctx->n_features * sizeof(BinMapper));

    printf("\nCreating bin mappers...\n");

    uint32_t direct_encoded = 0, bit2_encoded = 0, bit4_encoded = 0, binned_encoded = 0;

    for (uint32_t f = 0; f < ctx->n_features; f++) {
        float *feature_col = dataset->feature_columns[f].ptr;
        uint8_t *bin_col = binned->bin_columns[f].ptr;
        BinMapper *mapper = &ctx->bin_mappers[f];

        float unique_values[MAX_BINS];
        uint32_t cardinality = detect_feature_cardinality(feature_col, ctx->n_samples,
                                                          unique_values, MAX_BINS);

        mapper->cardinality = cardinality;
        uint8_t encoding_type = determine_encoding_type(cardinality);
        mapper->encoding_type = encoding_type;

        metadata[f].cardinality = cardinality;
        metadata[f].encoding_type = encoding_type;

        if (encoding_type == ENCODING_DIRECT || encoding_type == ENCODING_BITS_2 || encoding_type == ENCODING_BITS_4) {
            metadata[f].bits_per_value = 8;
            metadata[f].n_bins = (uint8_t)cardinality;
            mapper->n_bins = cardinality;

            for (uint32_t i = 0; i < cardinality; i++) {
                mapper->unique_values[i] = unique_values[i];
            }

            for (uint32_t i = 0; i < ctx->n_samples; i++) {
                float value = feature_col[i];
                uint8_t idx = 0;
                for (uint32_t u = 0; u < cardinality; u++) {
                    if (fabsf(value - unique_values[u]) < 1e-6f) {
                        idx = (uint8_t)u;
                        break;
                    }
                }
                bin_col[i] = idx;
            }

            if (encoding_type == ENCODING_DIRECT) direct_encoded++;
            else if (encoding_type == ENCODING_BITS_2) bit2_encoded++;
            else bit4_encoded++;

        } else {
            metadata[f].bits_per_value = 8;

            float *sorted = (float*)malloc(ctx->n_samples * sizeof(float));
            memcpy(sorted, feature_col, ctx->n_samples * sizeof(float));
            qsort(sorted, ctx->n_samples, sizeof(float), compare_floats);

            uint32_t n_bins = MAX_BINS < ctx->n_samples ? MAX_BINS : ctx->n_samples;
            mapper->n_bins = n_bins;
            metadata[f].n_bins = (uint8_t)n_bins;

            for (uint32_t b = 0; b < n_bins; b++) {
                uint32_t idx = (uint32_t)((b + 1.0) * ctx->n_samples / n_bins) - 1;
                if (idx >= ctx->n_samples) idx = ctx->n_samples - 1;
                mapper->bin_upper_bounds[b] = sorted[idx];
            }

            free(sorted);

            for (uint32_t i = 0; i < ctx->n_samples; i++) {
                float value = feature_col[i];
                uint32_t bin = 0;
                for (uint32_t b = 0; b < mapper->n_bins; b++) {
                    if (value <= mapper->bin_upper_bounds[b]) {
                        bin = b;
                        break;
                    }
                }
                if (bin >= mapper->n_bins) bin = mapper->n_bins - 1;
                bin_col[i] = (uint8_t)bin;
            }
            binned_encoded++;
        }
    }

    printf("  Encoding: %u direct, %u 2-bit, %u 4-bit, %u binned\n",
           direct_encoded, bit2_encoded, bit4_encoded, binned_encoded);
}

// ============================================================================
// EFB: Exclusive Feature Bundling
// ============================================================================

// Count non-zero values for a feature (sparse detection)
static uint32_t count_nonzero(uint8_t *bin_col, uint32_t n_samples) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        if (bin_col[i] != 0) count++;
    }
    return count;
}

// Count conflicts between two features (both non-zero at same sample)
static uint32_t count_conflicts(uint8_t *bin_col_a, uint8_t *bin_col_b, uint32_t n_samples) {
    uint32_t conflicts = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        if (bin_col_a[i] != 0 && bin_col_b[i] != 0) {
            conflicts++;
        }
    }
    return conflicts;
}

// Greedy bundling algorithm
void init_efb(TrainContext *ctx) {
    BinnedDatasetHeader *binned = (BinnedDatasetHeader*)ctx->binned_dataset;
    EFBContext *efb = &ctx->efb;

    efb->enabled = 0;
    efb->bundles = NULL;
    efb->n_bundles = 0;
    efb->feature_to_bundle = NULL;
    efb->feature_to_offset = NULL;
    efb->bundled_bins = NULL;

    // Skip EFB for small feature counts
    if (ctx->n_features < EFB_MIN_FEATURES) {
        printf("EFB disabled (< %d features)\n", EFB_MIN_FEATURES);
        return;
    }

    uint32_t n_features = ctx->n_features;
    uint32_t n_samples = ctx->n_samples;
    uint32_t max_conflicts = (uint32_t)(EFB_CONFLICT_RATIO * n_samples);

    // Calculate sparsity for each feature
    uint32_t *nonzero_counts = malloc(n_features * sizeof(uint32_t));
    uint32_t sparse_count = 0;

    for (uint32_t f = 0; f < n_features; f++) {
        uint8_t *bin_col = binned->bin_columns[f].ptr;
        nonzero_counts[f] = count_nonzero(bin_col, n_samples);
        if (nonzero_counts[f] < n_samples * 0.5) {
            sparse_count++;
        }
    }

    // Skip if not enough sparse features
    if (sparse_count < 5) {
        printf("EFB disabled (only %u sparse features)\n", sparse_count);
        free(nonzero_counts);
        return;
    }

    // Allocate mapping arrays
    efb->feature_to_bundle = malloc(n_features * sizeof(uint32_t));
    efb->feature_to_offset = malloc(n_features * sizeof(uint32_t));

    for (uint32_t f = 0; f < n_features; f++) {
        efb->feature_to_bundle[f] = UINT32_MAX;  // Not bundled yet
    }

    // Greedy bundling: try to add each feature to existing bundle or create new one
    efb->bundles = malloc(n_features * sizeof(FeatureBundle));
    efb->n_bundles = 0;

    // Sort features by sparsity (sparser features first for better bundling)
    uint32_t *sorted_features = malloc(n_features * sizeof(uint32_t));
    for (uint32_t f = 0; f < n_features; f++) {
        sorted_features[f] = f;
    }

    // Simple insertion sort by nonzero count (ascending)
    for (uint32_t i = 1; i < n_features; i++) {
        uint32_t key = sorted_features[i];
        int32_t j = i - 1;
        while (j >= 0 && nonzero_counts[sorted_features[j]] > nonzero_counts[key]) {
            sorted_features[j + 1] = sorted_features[j];
            j--;
        }
        sorted_features[j + 1] = key;
    }

    for (uint32_t i = 0; i < n_features; i++) {
        uint32_t f = sorted_features[i];
        uint8_t *bin_col_f = binned->bin_columns[f].ptr;
        uint32_t f_bins = ctx->bin_mappers[f].n_bins;

        // Try to find existing bundle with low conflict
        int32_t best_bundle = -1;
        uint32_t min_conflict = UINT32_MAX;

        for (uint32_t b = 0; b < efb->n_bundles; b++) {
            FeatureBundle *bundle = &efb->bundles[b];

            // Skip if bundle is full or would exceed MAX_BINS
            if (bundle->n_features >= MAX_BUNDLE_SIZE) continue;
            if (bundle->total_bins + f_bins > MAX_BINS) continue;

            // Calculate total conflicts with all features in bundle
            uint32_t total_conflicts = 0;
            for (uint32_t j = 0; j < bundle->n_features; j++) {
                uint32_t other_f = bundle->feature_indices[j];
                uint8_t *bin_col_other = binned->bin_columns[other_f].ptr;
                total_conflicts += count_conflicts(bin_col_f, bin_col_other, n_samples);
                if (total_conflicts > max_conflicts) break;
            }

            if (total_conflicts <= max_conflicts && total_conflicts < min_conflict) {
                min_conflict = total_conflicts;
                best_bundle = b;
            }
        }

        if (best_bundle >= 0) {
            // Add to existing bundle
            FeatureBundle *bundle = &efb->bundles[best_bundle];
            uint32_t idx = bundle->n_features;
            bundle->feature_indices[idx] = f;
            bundle->bin_offsets[idx] = bundle->total_bins;
            bundle->n_features++;
            bundle->total_bins += f_bins;

            efb->feature_to_bundle[f] = best_bundle;
            efb->feature_to_offset[f] = bundle->bin_offsets[idx];
        } else {
            // Create new bundle
            uint32_t b = efb->n_bundles;
            FeatureBundle *bundle = &efb->bundles[b];
            bundle->feature_indices[0] = f;
            bundle->bin_offsets[0] = 0;
            bundle->n_features = 1;
            bundle->total_bins = f_bins;

            efb->feature_to_bundle[f] = b;
            efb->feature_to_offset[f] = 0;
            efb->n_bundles++;
        }
    }

    free(sorted_features);
    free(nonzero_counts);

    // Count multi-feature bundles
    uint32_t multi_bundles = 0;
    for (uint32_t b = 0; b < efb->n_bundles; b++) {
        if (efb->bundles[b].n_features > 1) {
            multi_bundles++;
        }
    }

    // Only enable if we actually bundled something
    if (multi_bundles == 0) {
        printf("EFB disabled (no features bundled)\n");
        free(efb->bundles);
        free(efb->feature_to_bundle);
        free(efb->feature_to_offset);
        efb->bundles = NULL;
        efb->feature_to_bundle = NULL;
        efb->feature_to_offset = NULL;
        efb->n_bundles = 0;
        return;
    }

    // Create merged bin values for each bundle
    efb->bundled_bins = malloc(efb->n_bundles * n_samples);

    for (uint32_t b = 0; b < efb->n_bundles; b++) {
        FeatureBundle *bundle = &efb->bundles[b];
        uint8_t *bundle_bins = efb->bundled_bins + b * n_samples;

        // Initialize to zero
        memset(bundle_bins, 0, n_samples);

        // Merge features: bin_value = original_bin + offset
        for (uint32_t j = 0; j < bundle->n_features; j++) {
            uint32_t f = bundle->feature_indices[j];
            uint32_t offset = bundle->bin_offsets[j];
            uint8_t *bin_col = binned->bin_columns[f].ptr;

            for (uint32_t i = 0; i < n_samples; i++) {
                if (bin_col[i] != 0) {
                    bundle_bins[i] = (uint8_t)(bin_col[i] + offset);
                }
            }
        }
    }

    efb->enabled = 1;
    printf("EFB enabled: %u features -> %u bundles (%u multi-feature bundles)\n",
           n_features, efb->n_bundles, multi_bundles);
}

void free_efb(TrainContext *ctx) {
    EFBContext *efb = &ctx->efb;
    if (efb->bundles) free(efb->bundles);
    if (efb->feature_to_bundle) free(efb->feature_to_bundle);
    if (efb->feature_to_offset) free(efb->feature_to_offset);
    if (efb->bundled_bins) free(efb->bundled_bins);
    efb->bundles = NULL;
    efb->feature_to_bundle = NULL;
    efb->feature_to_offset = NULL;
    efb->bundled_bins = NULL;
    efb->n_bundles = 0;
    efb->enabled = 0;
}

// Build histogram for a bundle
static void build_histogram_bundle(TrainContext *ctx, uint32_t bundle_idx,
                                   uint32_t *samples, uint32_t n_samples, Histogram *hist) {
    EFBContext *efb = &ctx->efb;
    uint8_t *bundle_bins = efb->bundled_bins + bundle_idx * ctx->n_samples;

    memset(hist, 0, sizeof(Histogram));

    float *gradients = ctx->gradients;
    float *hessians = ctx->hessians;
    float *sample_weights = ctx->sample_weights;

    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t idx = samples[i];
        uint8_t bin = bundle_bins[idx];
        float weight = sample_weights[idx];

        hist->sum_gradients[bin] += gradients[idx] * weight;
        hist->sum_hessians[bin] += hessians[idx] * weight;
        hist->counts[bin]++;
    }
}

// Find best split for a bundle
static void find_best_split_for_bundle(TrainContext *ctx, uint32_t bundle_idx,
                                       Histogram *hist, SplitInfo *best_split) {
    EFBContext *efb = &ctx->efb;
    FeatureBundle *bundle = &efb->bundles[bundle_idx];

    uint32_t total_bins = bundle->total_bins;

    double total_grad = 0;
    double total_hess = 0;

    for (uint32_t b = 0; b < total_bins; b++) {
        total_grad += hist->sum_gradients[b];
        total_hess += hist->sum_hessians[b];
    }

    if (total_hess < 1e-6) return;

    // For each feature in bundle, try splits
    for (uint32_t j = 0; j < bundle->n_features; j++) {
        uint32_t f = bundle->feature_indices[j];
        uint32_t offset = bundle->bin_offsets[j];
        BinMapper *mapper = &ctx->bin_mappers[f];
        uint32_t n_bins = mapper->n_bins;

        double left_grad = 0;
        double left_hess = 0;

        // Accumulate bins 0..offset-1 (other features' bins before this one)
        for (uint32_t b = 0; b < offset; b++) {
            left_grad += hist->sum_gradients[b];
            left_hess += hist->sum_hessians[b];
        }

        // Try splits within this feature's bin range
        for (uint32_t b = 0; b < n_bins - 1; b++) {
            uint32_t actual_bin = offset + b;
            left_grad += hist->sum_gradients[actual_bin];
            left_hess += hist->sum_hessians[actual_bin];

            double right_grad = total_grad - left_grad;
            double right_hess = total_hess - left_hess;

            if (left_hess < 1e-6 || right_hess < 1e-6) continue;

            double gain = 0.5 * (left_grad * left_grad / left_hess +
                                 right_grad * right_grad / right_hess -
                                 total_grad * total_grad / total_hess);

            if (gain > best_split->gain) {
                best_split->gain = gain;
                best_split->feature_idx = f;
                best_split->bin_idx = b;

                if (mapper->encoding_type == ENCODING_DIRECT ||
                    mapper->encoding_type == ENCODING_BITS_2 ||
                    mapper->encoding_type == ENCODING_BITS_4) {
                    if (b < n_bins - 1) {
                        best_split->threshold = (mapper->unique_values[b] + mapper->unique_values[b + 1]) / 2.0f;
                    } else {
                        best_split->threshold = mapper->unique_values[b];
                    }
                } else {
                    best_split->threshold = mapper->bin_upper_bounds[b];
                }

                best_split->left_weight = -left_grad / left_hess;
                best_split->right_weight = -right_grad / right_hess;
            }
        }

        // Add last bin to left_grad/hess for next feature
        left_grad += hist->sum_gradients[offset + n_bins - 1];
        left_hess += hist->sum_hessians[offset + n_bins - 1];
    }
}

// Build histogram for a feature
void build_histogram(TrainContext *ctx, uint32_t feature_idx,
                     uint32_t *samples, uint32_t n_samples, Histogram *hist) {
    BinnedDatasetHeader *binned = (BinnedDatasetHeader*)ctx->binned_dataset;
    uint8_t *bin_col = binned->bin_columns[feature_idx].ptr;

    memset(hist, 0, sizeof(Histogram));

    float *gradients = ctx->gradients;
    float *hessians  = ctx->hessians;
    float *sample_weights = ctx->sample_weights;
    float *sum_gradients = hist->sum_gradients;
    float *sum_hessians = hist->sum_hessians;
    uint32_t *counts = hist->counts;

    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t idx = samples[i];
        uint8_t bin = bin_col[idx];
        float weight = sample_weights[idx];

        sum_gradients[bin] += gradients[idx] * weight;
        sum_hessians[bin] += hessians[idx] * weight;
        counts[bin]++;
    }
}

// Find best split for a feature
static void find_best_split_for_feature(TrainContext *ctx, uint32_t feature_idx,
                                        Histogram *hist, SplitInfo *best_split) {
    BinMapper *mapper = &ctx->bin_mappers[feature_idx];

    uint32_t n_bins = mapper->n_bins;
    uint8_t encoding_type = mapper->encoding_type;

    double total_grad = 0;
    double total_hess = 0;

    float *sum_gradients = hist->sum_gradients;
    float *sum_hessians = hist->sum_hessians;

    for (uint32_t b = 0; b < n_bins; b++) {
        total_grad += sum_gradients[b];
        total_hess += sum_hessians[b];
    }

    if (total_hess < 1e-6) return;

    double left_grad = 0;
    double left_hess = 0;

    for (uint32_t b = 0; b < n_bins - 1; b++) {
        left_grad += sum_gradients[b];
        left_hess += sum_hessians[b];

        double right_grad = total_grad - left_grad;
        double right_hess = total_hess - left_hess;

        if (left_hess < 1e-6 || right_hess < 1e-6) continue;

        double left_weight = -left_grad / left_hess;
        double right_weight = -right_grad / right_hess;

        double gain = 0.5 * (left_grad * left_grad / left_hess +
                            right_grad * right_grad / right_hess -
                            total_grad * total_grad / total_hess);

        if (gain > best_split->gain) {
            best_split->gain = gain;
            best_split->feature_idx = feature_idx;
            best_split->bin_idx = b;

            if (encoding_type == ENCODING_DIRECT ||
                encoding_type == ENCODING_BITS_2 ||
                encoding_type == ENCODING_BITS_4) {
                if (b < n_bins - 1) {
                    best_split->threshold = (mapper->unique_values[b] + mapper->unique_values[b + 1]) / 2.0f;
                } else {
                    best_split->threshold = mapper->unique_values[b];
                }
            } else {
                best_split->threshold = mapper->bin_upper_bounds[b];
            }

            best_split->left_weight = left_weight;
            best_split->right_weight = right_weight;
        }
    }
}

// Find best split across all features (or bundles if EFB enabled)
SplitInfo find_best_split(TrainContext *ctx, uint32_t *samples, uint32_t n_samples) {
    SplitInfo best_split = {0};
    best_split.gain = 0.0;

    EFBContext *efb = &ctx->efb;

    if (efb->enabled) {
        // Use bundled features
        for (uint32_t b = 0; b < efb->n_bundles; b++) {
            Histogram hist;
            build_histogram_bundle(ctx, b, samples, n_samples, &hist);
            find_best_split_for_bundle(ctx, b, &hist, &best_split);
        }
    } else {
        // Original per-feature search
        for (uint32_t f = 0; f < ctx->n_features; f++) {
            Histogram hist;
            build_histogram(ctx, f, samples, n_samples, &hist);
            find_best_split_for_feature(ctx, f, &hist, &best_split);
        }
    }

    return best_split;
}

// Heap operations for leaf-wise growth
static void heap_swap(LeafCandidate *a, LeafCandidate *b) {
    LeafCandidate temp = *a;
    *a = *b;
    *b = temp;
}

static void heap_sift_down(LeafCandidate *heap, uint32_t size, uint32_t i) {
    uint32_t largest = i;
    uint32_t left = 2 * i + 1;
    uint32_t right = 2 * i + 2;

    if (left < size && heap[left].split_info.gain > heap[largest].split_info.gain) {
        largest = left;
    }
    if (right < size && heap[right].split_info.gain > heap[largest].split_info.gain) {
        largest = right;
    }

    if (largest != i) {
        heap_swap(&heap[i], &heap[largest]);
        heap_sift_down(heap, size, largest);
    }
}

static void heap_sift_up(LeafCandidate *heap, uint32_t i) {
    while (i > 0) {
        uint32_t parent = (i - 1) / 2;
        if (heap[i].split_info.gain <= heap[parent].split_info.gain) {
            break;
        }
        heap_swap(&heap[i], &heap[parent]);
        i = parent;
    }
}

static void heap_push(LeafCandidate *heap, uint32_t *size, LeafCandidate candidate) {
    heap[*size] = candidate;
    heap_sift_up(heap, *size);
    (*size)++;
}

static LeafCandidate heap_pop(LeafCandidate *heap, uint32_t *size) {
    LeafCandidate result = heap[0];
    (*size)--;
    if (*size > 0) {
        heap[0] = heap[*size];
        heap_sift_down(heap, *size, 0);
    }
    return result;
}

// Partition samples based on split
static uint32_t partition_samples(TrainContext *ctx, uint32_t *samples, uint32_t n_samples,
                                  uint32_t feature_idx, uint8_t bin_idx) {
    BinnedDatasetHeader *binned = (BinnedDatasetHeader*)ctx->binned_dataset;
    uint8_t *feature_bins = binned->bin_columns[feature_idx].ptr;

    uint32_t left = 0;
    uint32_t right = n_samples - 1;

    while (left <= right) {
        while (left <= right && feature_bins[samples[left]] <= bin_idx) {
            left++;
        }
        while (left <= right && feature_bins[samples[right]] > bin_idx) {
            right--;
        }
        if (left < right) {
            uint32_t temp = samples[left];
            samples[left] = samples[right];
            samples[right] = temp;
            left++;
            right--;
        }
    }

    return left;
}

// Compute gradient/hessian sums
static void compute_node_stats(TrainContext *ctx, uint32_t *samples, uint32_t n_samples,
                               double *sum_grad, double *sum_hess) {
    float *gradients = ctx->gradients;
    float *hessians = ctx->hessians;
    *sum_grad = 0;
    *sum_hess = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t idx = samples[i];
        *sum_grad += gradients[idx];
        *sum_hess += hessians[idx];
    }
}

// Build tree using leaf-wise growth
uint32_t build_tree_leafwise(TrainContext *ctx, uint32_t *initial_samples, uint32_t n_samples) {
    uint32_t max_leaves = MAX_LEAVES;
    LeafCandidate *leaf_heap = malloc(sizeof(LeafCandidate) * max_leaves);
    uint32_t heap_size = 0;

    uint32_t *sample_pool = malloc(sizeof(uint32_t) * n_samples * max_leaves);
    uint32_t pool_offset = 0;

    LeafCandidate root;
    root.samples = sample_pool;
    memcpy(root.samples, initial_samples, sizeof(uint32_t) * n_samples);
    root.n_samples = n_samples;
    root.node_idx = add_tree_node(ctx->model, ctx->current_tree, NODE_LEAF);
    root.depth = 0;
    compute_node_stats(ctx, root.samples, root.n_samples, &root.sum_gradients, &root.sum_hessians);

    double root_weight = root.sum_hessians > 1e-6 ? -root.sum_gradients / root.sum_hessians : 0;
    set_leaf_node(ctx->model, ctx->current_tree, root.node_idx, (float)root_weight);

    pool_offset += n_samples;

    uint32_t n_sampled = goss_sample_selection(ctx, root.samples, root.n_samples);
    root.split_info = find_best_split(ctx, root.samples, n_sampled);

    if (root.split_info.gain > 0 && root.n_samples >= 2 * MIN_SAMPLES_LEAF) {
        heap_push(leaf_heap, &heap_size, root);
    }

    uint32_t num_leaves = 1;

    while (heap_size > 0 && num_leaves < max_leaves) {
        LeafCandidate best_leaf = heap_pop(leaf_heap, &heap_size);

        uint32_t n_left = partition_samples(ctx, best_leaf.samples, best_leaf.n_samples,
                                           best_leaf.split_info.feature_idx,
                                           best_leaf.split_info.bin_idx);
        uint32_t n_right = best_leaf.n_samples - n_left;

        uint32_t left_child_idx = add_tree_node(ctx->model, ctx->current_tree, NODE_LEAF);
        uint32_t right_child_idx = add_tree_node(ctx->model, ctx->current_tree, NODE_LEAF);

        set_split_node(ctx->model, ctx->current_tree, best_leaf.node_idx,
                      best_leaf.split_info.feature_idx, best_leaf.split_info.threshold,
                      left_child_idx, right_child_idx);

        num_leaves++;

        LeafCandidate left_leaf;
        left_leaf.samples = sample_pool + pool_offset;
        memcpy(left_leaf.samples, best_leaf.samples, sizeof(uint32_t) * n_left);
        left_leaf.n_samples = n_left;
        left_leaf.node_idx = left_child_idx;
        left_leaf.depth = best_leaf.depth + 1;
        compute_node_stats(ctx, left_leaf.samples, left_leaf.n_samples,
                          &left_leaf.sum_gradients, &left_leaf.sum_hessians);

        double left_weight = left_leaf.sum_hessians > 1e-6 ?
                            -left_leaf.sum_gradients / left_leaf.sum_hessians : 0;
        set_leaf_node(ctx->model, ctx->current_tree, left_child_idx, (float)left_weight);

        pool_offset += n_left;

        LeafCandidate right_leaf;
        right_leaf.samples = sample_pool + pool_offset;
        memcpy(right_leaf.samples, best_leaf.samples + n_left, sizeof(uint32_t) * n_right);
        right_leaf.n_samples = n_right;
        right_leaf.node_idx = right_child_idx;
        right_leaf.depth = best_leaf.depth + 1;
        compute_node_stats(ctx, right_leaf.samples, right_leaf.n_samples,
                          &right_leaf.sum_gradients, &right_leaf.sum_hessians);

        double right_weight = right_leaf.sum_hessians > 1e-6 ?
                             -right_leaf.sum_gradients / right_leaf.sum_hessians : 0;
        set_leaf_node(ctx->model, ctx->current_tree, right_child_idx, (float)right_weight);

        pool_offset += n_right;

        if (left_leaf.n_samples >= 2 * MIN_SAMPLES_LEAF && left_leaf.depth < MAX_DEPTH) {
            uint32_t n_sampled_left = goss_sample_selection(ctx, left_leaf.samples, left_leaf.n_samples);
            left_leaf.split_info = find_best_split(ctx, left_leaf.samples, n_sampled_left);
            if (left_leaf.split_info.gain > 0) {
                heap_push(leaf_heap, &heap_size, left_leaf);
            }
        }

        if (right_leaf.n_samples >= 2 * MIN_SAMPLES_LEAF && right_leaf.depth < MAX_DEPTH) {
            uint32_t n_sampled_right = goss_sample_selection(ctx, right_leaf.samples, right_leaf.n_samples);
            right_leaf.split_info = find_best_split(ctx, right_leaf.samples, n_sampled_right);
            if (right_leaf.split_info.gain > 0) {
                heap_push(leaf_heap, &heap_size, right_leaf);
            }
        }
    }

    free(leaf_heap);
    free(sample_pool);

    return root.node_idx;
}

// Predict with current model
float predict_sample(TrainContext *ctx, uint32_t sample_idx) {
    ColumnDatasetHeader *dataset = (ColumnDatasetHeader*)ctx->dataset;
    ColumnModelHeader *model = (ColumnModelHeader*)ctx->model;

    float prediction = model->base_score;

    for (uint32_t t = 0; t < ctx->current_tree; t++) {
        size_t tree_offset = t * model->max_nodes_per_tree;
        uint32_t node_idx = 0;

        while (model->node_types.ptr[tree_offset + node_idx] == NODE_SPLIT) {
            uint32_t feature_idx = model->feature_indices.ptr[tree_offset + node_idx];
            float threshold = model->thresholds.ptr[tree_offset + node_idx];
            float value = dataset->feature_columns[feature_idx].ptr[sample_idx];

            if (value <= threshold) {
                node_idx = model->left_children.ptr[tree_offset + node_idx];
            } else {
                node_idx = model->right_children.ptr[tree_offset + node_idx];
            }
        }

        prediction += model->learning_rate * model->leaf_values.ptr[tree_offset + node_idx];
    }

    return prediction;
}

// Predict contribution from a single tree
static float predict_tree_sample(TrainContext *ctx, uint32_t tree_id, uint32_t sample_idx) {
    ColumnDatasetHeader *dataset = (ColumnDatasetHeader*)ctx->dataset;
    ColumnModelHeader *model = (ColumnModelHeader*)ctx->model;
    size_t tree_offset = tree_id * model->max_nodes_per_tree;
    uint32_t node_idx = 0;
    while (model->node_types.ptr[tree_offset + node_idx] == NODE_SPLIT) {
        uint32_t feature_idx = model->feature_indices.ptr[tree_offset + node_idx];
        float threshold = model->thresholds.ptr[tree_offset + node_idx];
        float value = dataset->feature_columns[feature_idx].ptr[sample_idx];
        if (value <= threshold) {
            node_idx = model->left_children.ptr[tree_offset + node_idx];
        } else {
            node_idx = model->right_children.ptr[tree_offset + node_idx];
        }
    }
    return model->leaf_values.ptr[tree_offset + node_idx];
}

static void update_predictions_with_tree(TrainContext *ctx, uint32_t tree_id) {
    ColumnModelHeader *model = (ColumnModelHeader*)ctx->model;
    for (uint32_t i = 0; i < ctx->n_samples; i++) {
        float leaf_value = predict_tree_sample(ctx, tree_id, i);
        ctx->predictions[i] += model->learning_rate * leaf_value;
    }
}

// Train GBM model
void train_gbm(TrainContext *ctx, uint32_t n_trees, float learning_rate) {
    ColumnDatasetHeader *dataset = (ColumnDatasetHeader*)ctx->dataset;
    ColumnModelHeader *model = (ColumnModelHeader*)ctx->model;

    model->learning_rate = learning_rate;
    model->base_score = 0.0f;
    model->n_features = ctx->n_features;

    float *labels = dataset->labels.ptr;
    double sum = 0;
    for (uint32_t i = 0; i < ctx->n_samples; i++) {
        sum += labels[i];
    }
    model->base_score = sum / ctx->n_samples;

    for (uint32_t i = 0; i < ctx->n_samples; i++) {
        ctx->predictions[i] = model->base_score;
    }

    printf("\nTraining %u trees (lr=%.3f, base=%.2f)\n", n_trees, learning_rate, model->base_score);

    ctx->use_goss = (ctx->n_samples >= 1000) ? 1 : 0;
    ctx->goss_alpha = 0.2f;
    ctx->goss_beta = 0.1f;

    if (ctx->use_goss) {
        printf("GOSS enabled (alpha=%.2f, beta=%.2f)\n", ctx->goss_alpha, ctx->goss_beta);
    }

    ctx->sample_buffer = (uint32_t*)malloc(ctx->n_samples * sizeof(uint32_t));
    ctx->sorted_indices = (uint32_t*)malloc(ctx->n_samples * sizeof(uint32_t));
    ctx->sample_weights = (float*)malloc(ctx->n_samples * sizeof(float));

    clock_t start = clock();

    // Pull frequently accessed pointers out of the loop
    float *predictions = ctx->predictions;
    float *gradients = ctx->gradients;
    float *hessians = ctx->hessians;
    uint32_t *sample_buffer = ctx->sample_buffer;
    uint32_t n_samples = ctx->n_samples;

    for (uint32_t t = 0; t < n_trees; t++) {
        ctx->current_tree = t;

        for (uint32_t i = 0; i < n_samples; i++) {
            float residual = predictions[i] - labels[i];
            gradients[i] = residual;
            hessians[i] = 1.0f;
        }

        for (uint32_t i = 0; i < n_samples; i++) {
            sample_buffer[i] = i;
        }

        build_tree_leafwise(ctx, sample_buffer, n_samples);
        update_predictions_with_tree(ctx, t);

        double mse = 0;
        for (uint32_t i = 0; i < n_samples; i++) {
            float diff = predictions[i] - labels[i];
            mse += diff * diff;
        }
        double rmse = sqrt(mse / n_samples);

        if ((t + 1) % 10 == 0 || t == 0) {
            printf("  Tree %3u: RMSE=%.2f\n", t + 1, rmse);
        }
    }

    ctx->current_tree = n_trees;

    free(ctx->sample_buffer);
    free(ctx->sorted_indices);
    free(ctx->sample_weights);
    ctx->sample_buffer = NULL;
    ctx->sorted_indices = NULL;
    ctx->sample_weights = NULL;

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTraining completed in %.2f seconds\n", time_spent);
}

// Calculate metrics
void evaluate_model(TrainContext *ctx) {
    ColumnDatasetHeader *dataset = (ColumnDatasetHeader*)ctx->dataset;
    float *labels = dataset->labels.ptr;

    for (uint32_t i = 0; i < ctx->n_samples; i++) {
        ctx->predictions[i] = predict_sample(ctx, i);
    }

    double mse = 0, mae = 0;
    for (uint32_t i = 0; i < ctx->n_samples; i++) {
        float diff = ctx->predictions[i] - labels[i];
        mse += diff * diff;
        mae += fabs(diff);
    }

    double rmse = sqrt(mse / ctx->n_samples);
    mae /= ctx->n_samples;

    printf("\nFinal Metrics:\n");
    printf("  RMSE: %.2f\n", rmse);
    printf("  MAE:  %.2f\n", mae);
}
