#pragma once

#include "kv3d/kv/kv_block.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace kv3d {

/// Cold-tier index: snapshots serialized to disk (mmap or file I/O).
/// Optional for MVP — only engaged when a cold_store_path is configured.
class SnapshotIndex {
public:
    explicit SnapshotIndex(std::filesystem::path store_dir);
    ~SnapshotIndex();

    SnapshotIndex(const SnapshotIndex&) = delete;
    SnapshotIndex& operator=(const SnapshotIndex&) = delete;

    /// Persist a snapshot to disk. Serializes as a binary blob.
    bool save(const KVSnapshot& snapshot);

    /// Load a snapshot from disk by family_id.
    [[nodiscard]] std::shared_ptr<KVSnapshot> load(uint64_t family_id);

    /// Check if a snapshot is stored on disk.
    [[nodiscard]] bool has(uint64_t family_id) const;

    /// Remove a snapshot from disk.
    bool remove(uint64_t family_id);

    /// Total bytes on disk across all snapshots.
    [[nodiscard]] size_t disk_usage_bytes() const;

private:
    std::filesystem::path store_dir_;
    [[nodiscard]] std::filesystem::path snapshot_path(uint64_t family_id) const;
};

}  // namespace kv3d
