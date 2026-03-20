#include "kv3d/storage/gpu_cache.hpp"

#include <algorithm>
#include <cassert>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace kv3d {

// ── LRU cache backed by a doubly-linked list + hash map ──────────────────────
// In a CUDA build this would call cudaMalloc/cudaFree; for now it stays in
// host memory and serves as the hot-tier abstraction.

struct GPUCache::Impl {
    size_t capacity_bytes;
    size_t used_bytes{0};

    // LRU list: front = most recently used
    std::list<uint64_t> lru_order;
    std::unordered_map<uint64_t, std::pair<std::shared_ptr<KVSnapshot>,
                                           std::list<uint64_t>::iterator>> entries;
    std::mutex mutex;

    explicit Impl(size_t cap) : capacity_bytes(cap) {}

    void evict_until_fits(size_t needed) {
        while (used_bytes + needed > capacity_bytes && !entries.empty()) {
            const uint64_t victim = lru_order.back();
            lru_order.pop_back();
            auto it = entries.find(victim);
            if (it != entries.end()) {
                used_bytes -= it->second.first->byte_size();
                entries.erase(it);
            }
        }
    }
};

GPUCache::GPUCache(size_t capacity_bytes)
    : impl_(std::make_unique<Impl>(capacity_bytes)) {}

GPUCache::~GPUCache() = default;

bool GPUCache::insert(std::shared_ptr<KVSnapshot> snapshot) {
    if (!snapshot) return false;
    const size_t sz = snapshot->byte_size();
    if (sz > impl_->capacity_bytes) return false;

    std::lock_guard lock(impl_->mutex);
    impl_->evict_until_fits(sz);

    const uint64_t id = snapshot->family_id;
    // Remove existing entry if present
    auto existing = impl_->entries.find(id);
    if (existing != impl_->entries.end()) {
        impl_->used_bytes -= existing->second.first->byte_size();
        impl_->lru_order.erase(existing->second.second);
        impl_->entries.erase(existing);
    }

    impl_->lru_order.push_front(id);
    impl_->entries[id] = {std::move(snapshot), impl_->lru_order.begin()};
    impl_->used_bytes += sz;
    return true;
}

std::shared_ptr<KVSnapshot> GPUCache::get(uint64_t family_id) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->entries.find(family_id);
    if (it == impl_->entries.end()) return nullptr;

    // Move to front (most recently used)
    impl_->lru_order.erase(it->second.second);
    impl_->lru_order.push_front(family_id);
    it->second.second = impl_->lru_order.begin();
    return it->second.first;
}

bool GPUCache::evict(uint64_t family_id) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->entries.find(family_id);
    if (it == impl_->entries.end()) return false;
    impl_->used_bytes -= it->second.first->byte_size();
    impl_->lru_order.erase(it->second.second);
    impl_->entries.erase(it);
    return true;
}

size_t GPUCache::capacity_bytes() const noexcept { return impl_->capacity_bytes; }
size_t GPUCache::used_bytes() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->used_bytes;
}
size_t GPUCache::size() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->entries.size();
}

}  // namespace kv3d
