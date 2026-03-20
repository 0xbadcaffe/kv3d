#include "kv3d/storage/snapshot_index.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace kv3d {

// ── Binary serialisation format ───────────────────────────────────────────────
// File layout (little-endian):
//   [8]  magic   = 0x4B5633440000'0001  ("KV3D" + version 1)
//   [8]  family_id
//   [4]  token_count
//   [4]  n_layers  (= blocks.size())
//   per block:
//     [4] layer_idx
//     [4] token_offset
//     [4] token_count
//     [8] key_count   (floats)
//     [8] val_count   (floats)
//     [key_count * 4] key data
//     [val_count * 4] val data

static constexpr uint64_t MAGIC = 0x4B56334400000001ULL;

namespace {
template <typename T>
void write_pod(std::ofstream& f, T v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
T read_pod(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}
}  // namespace

SnapshotIndex::SnapshotIndex(std::filesystem::path store_dir)
    : store_dir_(std::move(store_dir)) {
    std::filesystem::create_directories(store_dir_);
}

SnapshotIndex::~SnapshotIndex() = default;

std::filesystem::path SnapshotIndex::snapshot_path(uint64_t family_id) const {
    // Use hex family_id as filename for easy enumeration.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx.kvsnap",
                  static_cast<unsigned long long>(family_id));
    return store_dir_ / buf;
}

bool SnapshotIndex::save(const KVSnapshot& snap) {
    const auto path = snapshot_path(snap.family_id);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;

    write_pod(f, MAGIC);
    write_pod(f, snap.family_id);
    write_pod(f, snap.token_count);
    write_pod(f, static_cast<uint32_t>(snap.blocks.size()));

    for (const auto& blk : snap.blocks) {
        write_pod(f, blk.layer_idx);
        write_pod(f, blk.token_offset);
        write_pod(f, blk.token_count);
        write_pod(f, static_cast<uint64_t>(blk.keys.size()));
        write_pod(f, static_cast<uint64_t>(blk.values.size()));
        f.write(reinterpret_cast<const char*>(blk.keys.data()),
                static_cast<std::streamsize>(blk.keys.size() * sizeof(float)));
        f.write(reinterpret_cast<const char*>(blk.values.data()),
                static_cast<std::streamsize>(blk.values.size() * sizeof(float)));
    }
    return f.good();
}

std::shared_ptr<KVSnapshot> SnapshotIndex::load(uint64_t family_id) {
    const auto path = snapshot_path(family_id);
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;

    const uint64_t magic = read_pod<uint64_t>(f);
    if (magic != MAGIC) return nullptr;

    auto snap = std::make_shared<KVSnapshot>();
    snap->family_id = read_pod<uint64_t>(f);
    snap->token_count = read_pod<uint32_t>(f);
    const uint32_t n_blocks = read_pod<uint32_t>(f);

    snap->blocks.resize(n_blocks);
    for (auto& blk : snap->blocks) {
        blk.layer_idx = read_pod<uint32_t>(f);
        blk.token_offset = read_pod<uint32_t>(f);
        blk.token_count = read_pod<uint32_t>(f);
        const uint64_t nk = read_pod<uint64_t>(f);
        const uint64_t nv = read_pod<uint64_t>(f);
        blk.keys.resize(nk);
        blk.values.resize(nv);
        f.read(reinterpret_cast<char*>(blk.keys.data()),
               static_cast<std::streamsize>(nk * sizeof(float)));
        f.read(reinterpret_cast<char*>(blk.values.data()),
               static_cast<std::streamsize>(nv * sizeof(float)));
    }

    return f.good() ? snap : nullptr;
}

bool SnapshotIndex::has(uint64_t family_id) const {
    return std::filesystem::exists(snapshot_path(family_id));
}

bool SnapshotIndex::remove(uint64_t family_id) {
    return std::filesystem::remove(snapshot_path(family_id));
}

size_t SnapshotIndex::disk_usage_bytes() const {
    size_t total = 0;
    for (const auto& entry : std::filesystem::directory_iterator(store_dir_)) {
        if (entry.path().extension() == ".kvsnap") {
            total += entry.file_size();
        }
    }
    return total;
}

}  // namespace kv3d
