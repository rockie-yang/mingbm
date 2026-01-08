/*
 * MinGBM Model - Memory-mapped tree storage
 */

#include "../header.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ALIGN_16(x) (((x) + 15) & ~15)

/*
 * Step 1: Calculate model size
 * 
 * Pre-allocate max_nodes_per_tree slots for each tree
 * Even if a tree uses fewer nodes, space is reserved
 * 
 * Layout:
 * - Header: sizeof(ColumnModelHeader) + n_trees * sizeof(uint64_t) * 6
 * - Node types: n_trees * max_nodes_per_tree * sizeof(uint8_t)
 * - Feature indices: n_trees * max_nodes_per_tree * sizeof(uint32_t)
 * - Thresholds: n_trees * max_nodes_per_tree * sizeof(float)
 * - Left children: n_trees * max_nodes_per_tree * sizeof(uint32_t)
 * - Right children: n_trees * max_nodes_per_tree * sizeof(uint32_t)
 * - Leaf values: n_trees * max_nodes_per_tree * sizeof(float)
 * - Tree sizes: n_trees * sizeof(uint32_t)
 * 
 * Example: 100 trees, max_depth=6 → max_nodes = 2^7-1 = 127
 *   Header: 32 + 100*8*6 = 4,832 bytes
 *   Node types: 100 * 127 * 1 = 12,700 bytes
 *   Feature indices: 100 * 127 * 4 = 50,800 bytes
 *   Thresholds: 100 * 127 * 4 = 50,800 bytes
 *   Left children: 100 * 127 * 4 = 50,800 bytes
 *   Right children: 100 * 127 * 4 = 50,800 bytes
 *   Leaf values: 100 * 127 * 4 = 50,800 bytes
 *   Tree sizes: 100 * 4 = 400 bytes
 *   Total: ~272 KB
 */
size_t calculate_model_size(uint32_t n_trees, uint32_t max_nodes_per_tree) {
    size_t header_size = sizeof(ColumnModelHeader) + 6 * n_trees * sizeof(uint64_t);
    header_size = ALIGN_16(header_size);
    
    size_t node_types_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint8_t));
    size_t feature_indices_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t thresholds_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(float));
    size_t left_children_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t right_children_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t leaf_values_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(float));
    size_t tree_sizes_size = ALIGN_16(n_trees * sizeof(uint32_t));
    
    size_t total = header_size + node_types_size + feature_indices_size + 
                   thresholds_size + left_children_size + right_children_size +
                   leaf_values_size + tree_sizes_size;
    
    printf("Model memory layout:\n");
    printf("  Header: %zu bytes\n", header_size);
    printf("  Node types: %zu bytes\n", node_types_size);
    printf("  Feature indices: %zu bytes\n", feature_indices_size);
    printf("  Thresholds: %zu bytes\n", thresholds_size);
    printf("  Left children: %zu bytes\n", left_children_size);
    printf("  Right children: %zu bytes\n", right_children_size);
    printf("  Leaf values: %zu bytes\n", leaf_values_size);
    printf("  Tree sizes: %zu bytes\n", tree_sizes_size);
    printf("  Total: %zu bytes (%.2f KB)\n", total, total / 1024.0);
    printf("  Trees: %u\n", n_trees);
    printf("  Max nodes per tree: %u\n", max_nodes_per_tree);
    
    return total;
}

/*
 * Step 2: Create model mmap and initialize header
 */
void* create_model_mmap(const char *filename, uint32_t n_trees, uint32_t max_nodes_per_tree) {
    size_t total_size = calculate_model_size(n_trees, max_nodes_per_tree);
    
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
    ColumnModelHeader *header = (ColumnModelHeader*)ptr;
    header->magic = MAGIC_MODEL;
    header->version = 1;
    header->n_trees = n_trees;
    header->max_nodes_per_tree = max_nodes_per_tree;
    header->n_features = 0;  // Will be set during training
    header->learning_rate = 0.1f;
    header->base_score = 0.0f;
    
    // Calculate offsets (save mode)
    size_t header_size = ALIGN_16(sizeof(ColumnModelHeader));
    
    // Base offset for each column
    size_t node_types_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint8_t));
    size_t feature_indices_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t thresholds_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(float));
    size_t left_children_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t right_children_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(uint32_t));
    size_t leaf_values_size = ALIGN_16(n_trees * max_nodes_per_tree * sizeof(float));
    
    header->node_types.offset = header_size;
    header->feature_indices.offset = header->node_types.offset + node_types_size;
    header->thresholds.offset = header->feature_indices.offset + feature_indices_size;
    header->left_children.offset = header->thresholds.offset + thresholds_size;
    header->right_children.offset = header->left_children.offset + left_children_size;
    header->leaf_values.offset = header->right_children.offset + right_children_size;
    header->tree_sizes.offset = header->leaf_values.offset + leaf_values_size;
    
    // Convert offsets to pointers (mmap mode)
    initialize_model_pointers(ptr);
    
    // Initialize tree sizes to 0 (no nodes used yet)
    uint32_t *tree_sizes = header->tree_sizes.ptr;
    for (uint32_t t = 0; t < n_trees; t++) {
        tree_sizes[t] = 0;
    }
    
    // Initialize all node types to UNUSED
    uint8_t *node_types = header->node_types.ptr;
    for (uint32_t i = 0; i < n_trees * max_nodes_per_tree; i++) {
        node_types[i] = NODE_UNUSED;
    }
    
    printf("Created model mmap: %s\n", filename);
    printf("  Mapped at: %p\n", ptr);
    printf("  Size: %zu bytes\n", total_size);
    
    return ptr;
}

/*
 * Step 3: Helper - Add node to tree
 * 
 * Finds next unused slot in tree's pre-allocated space
 * Returns node index within tree (0 to max_nodes_per_tree-1)
 */
int add_tree_node(void *model_ptr, uint32_t tree_id, uint8_t node_type) {
    ColumnModelHeader *header = (ColumnModelHeader*)model_ptr;
    uint32_t *tree_sizes = header->tree_sizes.ptr;
    
    if (tree_sizes[tree_id] >= header->max_nodes_per_tree) {
        fprintf(stderr, "Error: Tree %u full (max %u nodes)\n", 
                tree_id, header->max_nodes_per_tree);
        return -1;
    }
    
    uint32_t node_idx = tree_sizes[tree_id];
    tree_sizes[tree_id]++;
    
    // Set node type
    size_t tree_offset = tree_id * header->max_nodes_per_tree;
    header->node_types.ptr[tree_offset + node_idx] = node_type;
    
    return node_idx;
}

/*
 * Step 4: Helper - Set split node data
 */
void set_split_node(void *model_ptr, uint32_t tree_id, uint32_t node_idx,
                    uint32_t feature_idx, float threshold, 
                    uint32_t left_child, uint32_t right_child) {
    ColumnModelHeader *header = (ColumnModelHeader*)model_ptr;
    size_t tree_offset = tree_id * header->max_nodes_per_tree;
    size_t idx = tree_offset + node_idx;
    
    header->node_types.ptr[idx] = NODE_SPLIT;  // Convert to split node
    header->feature_indices.ptr[idx] = feature_idx;
    header->thresholds.ptr[idx] = threshold;
    header->left_children.ptr[idx] = left_child;
    header->right_children.ptr[idx] = right_child;
}

/*
 * Step 5: Helper - Set leaf node data
 */
void set_leaf_node(void *model_ptr, uint32_t tree_id, uint32_t node_idx, float value) {
    ColumnModelHeader *header = (ColumnModelHeader*)model_ptr;
    size_t tree_offset = tree_id * header->max_nodes_per_tree;
    header->leaf_values.ptr[tree_offset + node_idx] = value;
}

/*
 * Step 6: Helper - Print tree structure
 */
void print_tree(void *model_ptr, uint32_t tree_id) {
    ColumnModelHeader *header = (ColumnModelHeader*)model_ptr;
    uint32_t *tree_sizes = header->tree_sizes.ptr;
    
    if (tree_id >= header->n_trees) {
        fprintf(stderr, "Error: Invalid tree_id %u\n", tree_id);
        return;
    }
    
    size_t tree_offset = tree_id * header->max_nodes_per_tree;
    uint8_t *node_types = header->node_types.ptr + tree_offset;
    uint32_t *feature_indices = header->feature_indices.ptr + tree_offset;
    float *thresholds = header->thresholds.ptr + tree_offset;
    uint32_t *left_children = header->left_children.ptr + tree_offset;
    uint32_t *right_children = header->right_children.ptr + tree_offset;
    float *leaf_values = header->leaf_values.ptr + tree_offset;
    
    printf("Tree %u (%u/%u nodes used):\n", tree_id, tree_sizes[tree_id], header->max_nodes_per_tree);
    
    for (uint32_t i = 0; i < tree_sizes[tree_id]; i++) {
        printf("  Node %u: ", i);
        
        switch (node_types[i]) {
            case NODE_SPLIT:
                printf("SPLIT on feature %u > %.2f → L=%u, R=%u\n",
                       feature_indices[i], thresholds[i], 
                       left_children[i], right_children[i]);
                break;
            case NODE_LEAF:
                printf("LEAF value=%.4f\n", leaf_values[i]);
                break;
            case NODE_UNUSED:
                printf("UNUSED\n");
                break;
        }
    }
}

/*
 * Step 7: Helper - Predict using tree
 */
float predict_tree(void *model_ptr, uint32_t tree_id, float *features) {
    ColumnModelHeader *header = (ColumnModelHeader*)model_ptr;
    size_t tree_offset = tree_id * header->max_nodes_per_tree;
    
    uint8_t *node_types = header->node_types.ptr + tree_offset;
    uint32_t *feature_indices = header->feature_indices.ptr + tree_offset;
    float *thresholds = header->thresholds.ptr + tree_offset;
    uint32_t *left_children = header->left_children.ptr + tree_offset;
    uint32_t *right_children = header->right_children.ptr + tree_offset;
    float *leaf_values = header->leaf_values.ptr + tree_offset;
    
    uint32_t node_idx = 0;
    
    while (node_types[node_idx] == NODE_SPLIT) {
        uint32_t feature = feature_indices[node_idx];
        float threshold = thresholds[node_idx];
        
        if (features[feature] <= threshold) {
            node_idx = left_children[node_idx];
        } else {
            node_idx = right_children[node_idx];
        }
    }
    
    return leaf_values[node_idx];
}

/*
 * Step 8: Convert offsets to pointers after mmap
 */
void initialize_model_pointers(void *mmap_ptr) {
    ColumnModelHeader *header = (ColumnModelHeader*)mmap_ptr;
    uint8_t *base = (uint8_t*)mmap_ptr;
    
    // Save offsets before converting
    uint64_t node_types_offset = header->node_types.offset;
    uint64_t feature_indices_offset = header->feature_indices.offset;
    uint64_t thresholds_offset = header->thresholds.offset;
    uint64_t left_children_offset = header->left_children.offset;
    uint64_t right_children_offset = header->right_children.offset;
    uint64_t leaf_values_offset = header->leaf_values.offset;
    uint64_t tree_sizes_offset = header->tree_sizes.offset;
    
    // Convert offsets to pointers
    header->node_types.ptr = (uint8_t*)(base + node_types_offset);
    header->feature_indices.ptr = (uint32_t*)(base + feature_indices_offset);
    header->thresholds.ptr = (float*)(base + thresholds_offset);
    header->left_children.ptr = (uint32_t*)(base + left_children_offset);
    header->right_children.ptr = (uint32_t*)(base + right_children_offset);
    header->leaf_values.ptr = (float*)(base + leaf_values_offset);
    header->tree_sizes.ptr = (uint32_t*)(base + tree_sizes_offset);
}

/*
 * Step 9: Close model mmap
 */
void close_model_mmap(void *mmap_ptr, size_t size) {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) {
        msync(mmap_ptr, size, MS_SYNC);
        munmap(mmap_ptr, size);
        printf("Model mmap closed and synced\n");
    }
}
