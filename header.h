/*
 * MinGBM - Minimal Gradient Boosting Machine
 * Memory-mapped column-major storage for cache efficiency
 */

#ifndef HEADER_H
#define HEADER_H

#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>

// Training constants
#define MAX_BINS 256
#define MAX_DEPTH 6
#define MAX_LEAVES 31
#define MAX_BUNDLE_SIZE 8     // Max features per bundle
#define EFB_CONFLICT_RATIO 0.05  // Max conflict ratio for bundling

// ANSI Color codes
#define RESET   "\033[0m"
#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"
#define WHITE   "\033[1;37m"

// Magic numbers
#define MAGIC_DATASET 0xC01DA7A  // COL-DATA
#define MAGIC_MODEL   0xC01D017  // COL-MODEL

// Node types
#define NODE_UNUSED 0
#define NODE_SPLIT  1
#define NODE_LEAF   2

// ============================================================================
// BINNED DATASET LAYOUT (Adaptive encoding based on cardinality)
// ============================================================================

// Feature encoding types
#define ENCODING_DIRECT   0  // No binning needed, cardinality < 256, store as uint8_t
#define ENCODING_BITS_2   1  // 2-bit encoding (cardinality <= 4)
#define ENCODING_BITS_4   2  // 4-bit encoding (cardinality <= 16)
#define ENCODING_BINNED   3  // Standard binning (cardinality >= 256 or continuous)

/*
 * Feature metadata for adaptive encoding
 * Tracks cardinality and encoding strategy per feature
 */
typedef struct {
    uint32_t cardinality;    // Number of unique values
    uint8_t encoding_type;   // ENCODING_DIRECT, ENCODING_BITS_2, ENCODING_BITS_4, or ENCODING_BINNED
    uint8_t bits_per_value;  // Bits needed per value (2, 4, or 8)
    uint8_t n_bins;          // Number of bins (for binned encoding)
    uint8_t _padding;        // Alignment
} FeatureMetadata;

/*
 * Binned Dataset Header
 * Pre-computed bin/value indices for all features and samples
 * Uses adaptive encoding based on feature cardinality
 * 
 * Memory layout: column-major, one column per feature
 * Size varies by encoding type per feature
 */
typedef struct {
    uint32_t magic;          // Magic number: 0xB17DA7A
    uint32_t version;        // Format version: 2 (changed for adaptive encoding)
    uint32_t n_samples;      // Number of rows
    uint32_t n_features;     // Number of features
    uint32_t n_bins;         // Max bins per feature (usually 256)
    
    // Union: offset when saving, ptr after mmap
    union {
        uint64_t offset;     // Offset to metadata array
        FeatureMetadata *ptr; // Direct pointer to metadata
    } metadata;
    
    union {
        uint64_t offset;     // Offset to bin data start
        uint8_t *ptr;        // Direct pointer to bin data
    } bins;
    
    // Feature bin column offsets
    union {
        uint64_t offset;
        uint8_t *ptr;
    } bin_columns[0];        // Flexible array: [feature_idx][sample_idx]
    
} BinnedDatasetHeader;

// ============================================================================
// TRAINING TYPES
// ============================================================================

// Histogram for histogram-based splitting
typedef struct {
    float sum_gradients[MAX_BINS];
    float sum_hessians[MAX_BINS];
    uint32_t counts[MAX_BINS];
} Histogram;

// Binning info for a feature
typedef struct {
    float bin_upper_bounds[MAX_BINS];
    uint32_t n_bins;
    // For direct encoding: value to index mapping
    float unique_values[MAX_BINS];  // Sorted unique values
    uint32_t cardinality;            // Number of unique values
    uint8_t encoding_type;           // Type of encoding used
} BinMapper;

// EFB: Exclusive Feature Bundle
typedef struct {
    uint32_t feature_indices[MAX_BUNDLE_SIZE];  // Original feature indices in this bundle
    uint32_t bin_offsets[MAX_BUNDLE_SIZE];      // Bin offset for each feature in bundle
    uint32_t n_features;                         // Number of features in bundle
    uint32_t total_bins;                         // Total bins in merged histogram
} FeatureBundle;

// EFB context
typedef struct {
    FeatureBundle *bundles;      // Array of bundles
    uint32_t n_bundles;          // Number of bundles
    uint32_t *feature_to_bundle; // Map: feature_idx -> bundle_idx
    uint32_t *feature_to_offset; // Map: feature_idx -> bin_offset within bundle
    uint8_t *bundled_bins;       // Merged bin values: [bundle_idx][sample_idx]
    uint8_t enabled;             // Whether EFB is enabled
} EFBContext;

// Forward declaration
typedef struct TrainContext TrainContext;

// Split information
typedef struct {
    uint32_t feature_idx;
    uint8_t bin_idx;
    float threshold;
    double gain;
    double left_weight;
    double right_weight;
} SplitInfo;

// Leaf candidate for leaf-wise tree growth (max heap)
typedef struct {
    uint32_t *samples;        // Sample indices for this leaf
    uint32_t n_samples;       // Number of samples
    SplitInfo split_info;     // Best split for this leaf
    uint32_t node_idx;        // Node index in the tree
    uint32_t depth;           // Current depth
    double sum_gradients;     // Sum of gradients
    double sum_hessians;      // Sum of hessians
} LeafCandidate;

// Training context
struct TrainContext {
    void *dataset;        // Original float dataset
    void *binned_dataset; // Pre-computed bin indices (uint8_t)
    void *model;
    float *gradients;
    float *hessians;
    float *predictions;
    BinMapper *bin_mappers;
    uint32_t *sample_buffer;  // Pre-allocated buffer for sample indices
    uint32_t *sorted_indices; // For GOSS: indices sorted by gradient magnitude
    float *sample_weights;    // For GOSS: weights for sampled instances
    uint32_t n_samples;
    uint32_t n_features;
    uint32_t current_tree;
    // GOSS parameters
    float goss_alpha;         // Ratio of large gradient samples to keep
    float goss_beta;          // Ratio of small gradient samples to keep
    uint8_t use_goss;         // Whether to use GOSS
    // EFB context
    EFBContext efb;
};

// Split finding
SplitInfo find_best_split(TrainContext *ctx, uint32_t *samples, uint32_t n_samples);

// ============================================================================
// COLUMN DATASET LAYOUT
// ============================================================================

/*
 * Column Dataset Header
 * Describes the memory layout of column-major data
 * 
 * Example for house prices dataset:
 *   n_samples = 1460
 *   n_features = 79
 *   Total size = 1460 * 79 * 4 + metadata = ~461 KB
 */
typedef struct {
    uint32_t magic;          // Magic number: 0xC0LDATA
    uint32_t version;        // Format version: 1
    uint32_t n_samples;      // Number of rows (e.g., 1460)
    uint32_t n_features;     // Number of columns (e.g., 79)
    
    // Union: offset when saving, ptr after mmap
    union {
        uint64_t offset;     // Offset to label array (save mode)
        float *ptr;          // Direct pointer (mmap mode)
    } labels;
    
    union {
        uint64_t offset;     // Offset to feature columns start (save mode)
        float *ptr;          // Direct pointer (mmap mode)
    } features;
    
    // Feature column offsets (n_features entries)
    // Union: offsets when saving, pointers after mmap
    union {
        uint64_t offset;     // Offset for this feature column
        float *ptr;          // Direct pointer to this feature column
    } feature_columns[0];    // Flexible array member
    
} ColumnDatasetHeader;

/*
 * Memory layout of column dataset file:
 * 
 * [ColumnDatasetHeader]
 *   - magic, version, n_samples, n_features
 *   - labels_offset, features_offset
 *   - feature_offsets[0..n_features-1]
 * 
 * [Labels Array] (n_samples * sizeof(float))
 *   labels[0], labels[1], ..., labels[n_samples-1]
 * 
 * [Feature Column 0] (n_samples * sizeof(float))
 *   feature0[0], feature0[1], ..., feature0[n_samples-1]
 * 
 * [Feature Column 1] (n_samples * sizeof(float))
 *   feature1[0], feature1[1], ..., feature1[n_samples-1]
 * 
 * ...
 * 
 * [Feature Column n_features-1]
 *   feature_N[0], feature_N[1], ..., feature_N[n_samples-1]
 */

// ============================================================================
// MODEL LAYOUT
// ============================================================================

/*
 * Tree Node in column-major format
 * All nodes pre-allocated, may have holes
 * 
 * Example: max_leaves=127 for depth-6 tree
 *   Total nodes = 2^7 - 1 = 127
 *   Some nodes unused if tree is not full
 */
typedef struct {
    uint32_t magic;           // Magic number: 0xC0LMODEL
    uint32_t version;         // Format version: 1
    uint32_t n_trees;         // Number of trees (e.g., 50)
    uint32_t n_features;      // Number of features (e.g., 79)
    uint32_t max_nodes_per_tree; // Pre-allocated slots per tree (e.g., 127)
    
    float learning_rate;      // Learning rate (e.g., 0.1)
    float base_score;         // Initial prediction (e.g., 180590.0)
    
    // Unions: offsets when saving, pointers after mmap
    union {
        uint64_t offset;
        uint8_t *ptr;
    } node_types;            // uint8_t[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        uint32_t *ptr;
    } feature_indices;       // uint32_t[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        float *ptr;
    } thresholds;            // float[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        uint32_t *ptr;
    } left_children;         // uint32_t[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        uint32_t *ptr;
    } right_children;        // uint32_t[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        float *ptr;
    } leaf_values;           // float[n_trees * max_nodes_per_tree]
    
    union {
        uint64_t offset;
        uint32_t *ptr;
    } tree_sizes;            // uint32_t[n_trees] - actual nodes used
    
} ColumnModelHeader;

/*
 * Memory layout of model file:
 * 
 * [ColumnModelHeader]
 *   - magic, version, n_trees, n_features, max_nodes_per_tree
 *   - learning_rate, base_score
 *   - all column offsets
 * 
 * [Node Types Column] (n_trees * max_nodes_per_tree bytes)
 *   node_types[0], node_types[1], ..., node_types[total_nodes-1]
 *   0 = leaf, 1 = internal
 * 
 * [Feature Indices Column] (n_trees * max_nodes_per_tree * 4 bytes)
 *   feature_indices[0], feature_indices[1], ...
 * 
 * [Thresholds Column] (n_trees * max_nodes_per_tree * 4 bytes)
 *   thresholds[0], thresholds[1], ...
 * 
 * [Left Child Column] (n_trees * max_nodes_per_tree * 4 bytes)
 *   left_child[0], left_child[1], ...
 * 
 * [Right Child Column] (n_trees * max_nodes_per_tree * 4 bytes)
 *   right_child[0], right_child[1], ...
 * 
 * [Leaf Values Column] (n_trees * max_nodes_per_tree * 4 bytes)
 *   leaf_values[0], leaf_values[1], ...
 * 
 * [Tree Sizes] (n_trees * 4 bytes)
 *   tree_sizes[0], tree_sizes[1], ..., tree_sizes[n_trees-1]
 */

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Dataset functions
int detect_csv_format(const char *filename, uint32_t *n_samples, uint32_t *n_features);
size_t calculate_dataset_size(uint32_t n_samples, uint32_t n_features);
void* create_dataset_mmap(const char *filename, uint32_t n_samples, uint32_t n_features);
int load_csv_to_mmap(const char *csv_file, void *mmap_ptr);
void close_dataset_mmap(void *mmap_ptr, size_t size);

// Model functions
size_t calculate_model_size(uint32_t n_trees, uint32_t max_nodes_per_tree);
void* create_model_mmap(const char *filename, uint32_t n_trees, uint32_t max_nodes_per_tree);
int add_tree_node(void *model_ptr, uint32_t tree_id, uint8_t node_type);
void set_split_node(void *model_ptr, uint32_t tree_id, uint32_t node_idx,
                    uint32_t feature_idx, float threshold, 
                    uint32_t left_child, uint32_t right_child);
void set_leaf_node(void *model_ptr, uint32_t tree_id, uint32_t node_idx, float value);
void print_tree(void *model_ptr, uint32_t tree_id);
float predict_tree(void *model_ptr, uint32_t tree_id, float *features);
void close_model_mmap(void *mmap_ptr, size_t size);

// Helper function to convert offsets to pointers after mmap
void initialize_dataset_pointers(void *mmap_ptr);
void initialize_model_pointers(void *mmap_ptr);

// Binned dataset functions
size_t calculate_binned_size(uint32_t n_samples, uint32_t n_features);
void* create_binned_mmap(const char *filename, uint32_t n_samples, uint32_t n_features, uint32_t n_bins);
void initialize_binned_pointers(void *mmap_ptr);
void close_binned_mmap(void *mmap_ptr, size_t size);
uint32_t detect_feature_cardinality(float *feature_col, uint32_t n_samples, 
                                    float *unique_values, uint32_t max_unique);
uint8_t determine_encoding_type(uint32_t cardinality);

// Training functions
void create_bin_mappers(TrainContext *ctx);
void train_gbm(TrainContext *ctx, uint32_t n_trees, float learning_rate);
void evaluate_model(TrainContext *ctx);

// EFB functions
void init_efb(TrainContext *ctx);
void free_efb(TrainContext *ctx);

// Utility functions
void print_color(const char *color, const char *format, ...);
void print_help(const char *program_name);

#endif // HEADER_H
