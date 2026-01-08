/*
 * MinGBM Dataset - Memory-mapped column-major storage
 */

#include "../header.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define ALIGN_16(x) (((x) + 15) & ~15)

/*
 * Step 1: Detect CSV format
 * 
 * Example: house-prices-advanced-regression-techniques/train.csv
 * - Header line: Id,MSSubClass,MSZoning,...,SalePrice
 * - Data lines: 1,60,RL,...,208500
 * 
 * Returns: n_samples=1460, n_features=79 (excluding Id and SalePrice)
 */
int detect_csv_format(const char *filename, uint32_t *n_samples, uint32_t *n_features) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }
    
    char line[8192];
    
    // Read header to count columns
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }
    
    uint32_t n_cols = 1;
    for (char *p = line; *p; p++) {
        if (*p == ',') n_cols++;
    }
    
    // n_features = total_cols - Id - SalePrice
    *n_features = n_cols - 2;
    
    // Count data rows
    *n_samples = 0;
    while (fgets(line, sizeof(line), fp)) {
        (*n_samples)++;
    }
    
    fclose(fp);
    
    printf("Detected CSV format:\n");
    printf("  Samples: %u\n", *n_samples);
    printf("  Features: %u\n", *n_features);
    printf("  Total columns: %u (Id + %u features + Label)\n", n_cols, *n_features);
    
    return 0;
}

/*
 * Step 2: Calculate total size needed
 * 
 * Layout:
 * - Header: sizeof(ColumnDatasetHeader) + n_features * sizeof(uint64_t)
 * - Labels: n_samples * sizeof(float)
 * - Each feature column: n_samples * sizeof(float)
 * 
 * Total = header + labels + (n_features * n_samples * sizeof(float))
 * 
 * Example: 1460 samples, 79 features
 *   Header: 32 + 79*8 = 664 bytes
 *   Labels: 1460 * 4 = 5,840 bytes
 *   Features: 79 * 1460 * 4 = 461,360 bytes
 *   Total: ~468 KB
 */
size_t calculate_dataset_size(uint32_t n_samples, uint32_t n_features) {
    size_t header_size = sizeof(ColumnDatasetHeader) + n_features * sizeof(uint64_t);
    header_size = ALIGN_16(header_size);
    
    size_t labels_size = ALIGN_16(n_samples * sizeof(float));
    size_t features_size = n_features * ALIGN_16(n_samples * sizeof(float));
    
    size_t total = header_size + labels_size + features_size;
    
    printf("Dataset memory layout:\n");
    printf("  Header: %zu bytes\n", header_size);
    printf("  Labels: %zu bytes\n", labels_size);
    printf("  Features: %zu bytes (%u columns)\n", features_size, n_features);
    printf("  Total: %zu bytes (%.2f KB)\n", total, total / 1024.0);
    
    return total;
}

/*
 * Step 3: Create mmap file and initialize header
 * 
 * Creates a new file, resizes it, and maps into memory
 */
void* create_dataset_mmap(const char *filename, uint32_t n_samples, uint32_t n_features) {
    size_t total_size = calculate_dataset_size(n_samples, n_features);
    
    // Create file
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return NULL;
    }
    
    // Resize file
    if (ftruncate(fd, total_size) < 0) {
        perror("ftruncate");
        close(fd);
        return NULL;
    }
    
    // Map into memory
    void *ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return NULL;
    }
    
    close(fd);  // Can close fd, mmap keeps reference
    
    // Initialize header
    ColumnDatasetHeader *header = (ColumnDatasetHeader*)ptr;
    header->magic = MAGIC_DATASET;
    header->version = 1;
    header->n_samples = n_samples;
    header->n_features = n_features;
    
    // Calculate offsets (save mode)
    size_t header_size = ALIGN_16(sizeof(ColumnDatasetHeader) + n_features * sizeof(uint64_t) * 2);
    header->labels.offset = header_size;
    header->features.offset = header->labels.offset + ALIGN_16(n_samples * sizeof(float));
    
    // Set feature column offsets
    for (uint32_t i = 0; i < n_features; i++) {
        header->feature_columns[i].offset = header->features.offset + i * ALIGN_16(n_samples * sizeof(float));
    }
    
    // Convert offsets to pointers (mmap mode)
    initialize_dataset_pointers(ptr);
    
    printf("Created dataset mmap: %s\n", filename);
    printf("  Mapped at: %p\n", ptr);
    printf("  Size: %zu bytes\n", total_size);
    
    return ptr;
}

/*
 * Step 4: Load CSV data into mmap
 * 
 * Reads CSV and stores column-by-column
 * Format: Id,Feature1,Feature2,...,FeatureN,SalePrice
 */
int load_csv_to_mmap(const char *csv_file, void *mmap_ptr) {
    FILE *fp = fopen(csv_file, "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }
    
    ColumnDatasetHeader *header = (ColumnDatasetHeader*)mmap_ptr;
    float *labels = header->labels.ptr;
    
    // Get feature column pointers (already initialized)
    float **features = (float**)malloc(sizeof(float*) * header->n_features);
    for (uint32_t i = 0; i < header->n_features; i++) {
        features[i] = header->feature_columns[i].ptr;
    }
    
    // Skip header line
    char line[8192];
    fgets(line, sizeof(line), fp);
    
    // Read data rows
    uint32_t row = 0;
    printf("Loading CSV data...\n");
    
    while (fgets(line, sizeof(line), fp) && row < header->n_samples) {
        char *token = strtok(line, ",");
        int col = 0;
        
        while (token != NULL) {
            if (col == 0) {
                // Skip Id column
            } else if (col <= (int)header->n_features) {
                // Feature columns
                float value = atof(token);
                if (strlen(token) == 0 || strcmp(token, "NA") == 0) {
                    value = 0.0f;  // Handle missing values
                }
                features[col - 1][row] = value;
            } else {
                // Label column (last)
                labels[row] = atof(token);
            }
            token = strtok(NULL, ",");
            col++;
        }
        
        row++;
        if (row % 100 == 0) {
            printf("  Loaded %u/%u rows\r", row, header->n_samples);
            fflush(stdout);
        }
    }
    
    printf("\nLoaded %u rows into mmap\n", row);
    
    // Verify first few values
    printf("Sample data (first 3 samples):\n");
    printf("  Labels: [%.2f, %.2f, %.2f]\n", labels[0], labels[1], labels[2]);
    printf("  Feature 0: [%.2f, %.2f, %.2f]\n", features[0][0], features[0][1], features[0][2]);
    printf("  Feature 1: [%.2f, %.2f, %.2f]\n", features[1][0], features[1][1], features[1][2]);
    
    free(features);
    fclose(fp);
    
    return 0;
}

/*
 * Step 5: Convert offsets to pointers after mmap
 */
void initialize_dataset_pointers(void *mmap_ptr) {
    ColumnDatasetHeader *header = (ColumnDatasetHeader*)mmap_ptr;
    uint8_t *base = (uint8_t*)mmap_ptr;
    
    // Convert offsets to pointers
    uint64_t labels_offset = header->labels.offset;
    uint64_t features_offset = header->features.offset;
    
    header->labels.ptr = (float*)(base + labels_offset);
    header->features.ptr = (float*)(base + features_offset);
    
    // Convert feature column offsets to pointers
    for (uint32_t i = 0; i < header->n_features; i++) {
        uint64_t col_offset = header->feature_columns[i].offset;
        header->feature_columns[i].ptr = (float*)(base + col_offset);
    }
}

/*
 * Step 6: Close mmap and sync to disk
 */
void close_dataset_mmap(void *mmap_ptr, size_t size) {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) {
        msync(mmap_ptr, size, MS_SYNC);  // Sync to disk
        munmap(mmap_ptr, size);
        printf("Dataset mmap closed and synced\n");
    }
}

/*
 * Binned Dataset Implementation with Adaptive Encoding
 * Pre-compute and store bin indices for all features
 */

#define MAGIC_BINNED 0xB17DA7A  // BIN-DATA

/*
 * Detect cardinality and unique values for a feature column
 * Returns: number of unique values found
 */
uint32_t detect_feature_cardinality(float *feature_col, uint32_t n_samples, 
                                    float *unique_values, uint32_t max_unique) {
    uint32_t n_unique = 0;
    
    // Collect unique values
    for (uint32_t i = 0; i < n_samples && n_unique < max_unique; i++) {
        float value = feature_col[i];
        
        // Check if value already exists
        uint32_t j;
        for (j = 0; j < n_unique; j++) {
            if (fabsf(unique_values[j] - value) < 1e-6f) {
                break;
            }
        }
        
        // Add new unique value
        if (j == n_unique) {
            unique_values[n_unique++] = value;
            if (n_unique >= max_unique) {
                return max_unique;  // Early exit if too many unique values
            }
        }
    }
    
    // Sort unique values for efficient lookup
    for (uint32_t i = 0; i < n_unique - 1; i++) {
        for (uint32_t j = i + 1; j < n_unique; j++) {
            if (unique_values[i] > unique_values[j]) {
                float temp = unique_values[i];
                unique_values[i] = unique_values[j];
                unique_values[j] = temp;
            }
        }
    }
    
    return n_unique;
}

/*
 * Determine optimal encoding type based on cardinality
 */
uint8_t determine_encoding_type(uint32_t cardinality) {
    if (cardinality <= 4) {
        return ENCODING_BITS_2;
    } else if (cardinality <= 16) {
        return ENCODING_BITS_4;
    } else if (cardinality < 256) {
        return ENCODING_DIRECT;
    } else {
        return ENCODING_BINNED;
    }
}

size_t calculate_binned_size(uint32_t n_samples, uint32_t n_features) {
    // Conservative estimate: assume all features use 1 byte per sample
    // Actual size will be computed during encoding
    size_t header_size = ALIGN_16(sizeof(BinnedDatasetHeader) + 
                                   n_features * sizeof(uint64_t) * 2 + 
                                   n_features * sizeof(FeatureMetadata));
    size_t bin_data_size = n_features * ALIGN_16(n_samples * sizeof(uint8_t));
    size_t total = header_size + bin_data_size;
    
    printf("Binned dataset layout (estimated):\n");
    printf("  Header: %zu bytes\n", header_size);
    printf("  Bin data: %zu bytes (%u features × %u samples)\n", bin_data_size, n_features, n_samples);
    printf("  Total: %zu bytes (%.2f KB)\n", total, total / 1024.0);
    
    return total;
}

void* create_binned_mmap(const char *filename, uint32_t n_samples, uint32_t n_features, uint32_t n_bins) {
    size_t total_size = calculate_binned_size(n_samples, n_features);
    
    // Create file
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return NULL;
    }
    
    // Resize file
    if (ftruncate(fd, total_size) < 0) {
        perror("ftruncate");
        close(fd);
        return NULL;
    }
    
    // Map into memory
    void *ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return NULL;
    }
    
    close(fd);
    
    // Initialize header
    BinnedDatasetHeader *header = (BinnedDatasetHeader*)ptr;
    header->magic = MAGIC_BINNED;
    header->version = 2;  // Version 2 for adaptive encoding
    header->n_samples = n_samples;
    header->n_features = n_features;
    header->n_bins = n_bins;
    
    // Calculate offsets
    size_t header_size = ALIGN_16(sizeof(BinnedDatasetHeader) + n_features * sizeof(uint64_t) * 2);
    
    // Metadata comes first
    header->metadata.offset = header_size;
    size_t metadata_size = ALIGN_16(n_features * sizeof(FeatureMetadata));
    
    // Then bin data
    header->bins.offset = header->metadata.offset + metadata_size;
    
    // Set feature bin column offsets (conservative allocation, 1 byte per sample)
    for (uint32_t i = 0; i < n_features; i++) {
        header->bin_columns[i].offset = header->bins.offset + i * ALIGN_16(n_samples * sizeof(uint8_t));
    }
    
    // Convert offsets to pointers
    initialize_binned_pointers(ptr);
    
    printf("Created binned dataset mmap: %s\n", filename);
    printf("  Mapped at: %p\n", ptr);
    printf("  Size: %zu bytes\n", total_size);
    
    return ptr;
}

void initialize_binned_pointers(void *mmap_ptr) {
    BinnedDatasetHeader *header = (BinnedDatasetHeader*)mmap_ptr;
    uint8_t *base = (uint8_t*)mmap_ptr;
    
    // Convert metadata offset to pointer
    uint64_t metadata_offset = header->metadata.offset;
    header->metadata.ptr = (FeatureMetadata*)(base + metadata_offset);
    
    // Convert offsets to pointers
    uint64_t bins_offset = header->bins.offset;
    header->bins.ptr = base + bins_offset;
    
    // Convert bin column offsets to pointers
    for (uint32_t i = 0; i < header->n_features; i++) {
        uint64_t col_offset = header->bin_columns[i].offset;
        header->bin_columns[i].ptr = base + col_offset;
    }
}

void close_binned_mmap(void *mmap_ptr, size_t size) {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) {
        msync(mmap_ptr, size, MS_SYNC);
        munmap(mmap_ptr, size);
        printf("Binned dataset mmap closed and synced\n");
    }
}
