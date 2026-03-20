#include "kv3d/kv/prefix_store.hpp"

#include <algorithm>
#include <limits>

namespace kv3d {

void PrefixStore::insert(std::shared_ptr<KVSnapshot> snapshot) {
    if (!snapshot) return;
    const uint64_t id = snapshot->family_id;

    std::lock_guard lock(mutex_);
    store_[id] = std::move(snapshot);
    access_epoch_[id] = ++epoch_;
}

std::shared_ptr<KVSnapshot> PrefixStore::get(uint64_t family_id) const {
    std::lock_guard lock(mutex_);
    auto it = store_.find(family_id);
    if (it == store_.end()) return nullptr;
    // Update LRU epoch (const-cast is safe: epoch tracking is logically non-const)
    const_cast<PrefixStore*>(this)->access_epoch_[family_id] = ++const_cast<PrefixStore*>(this)->epoch_;
    return it->second;
}

bool PrefixStore::contains(uint64_t family_id) const {
    std::lock_guard lock(mutex_);
    return store_.contains(family_id);
}

bool PrefixStore::evict(uint64_t family_id) {
    std::lock_guard lock(mutex_);
    access_epoch_.erase(family_id);
    return store_.erase(family_id) > 0;
}

size_t PrefixStore::size() const {
    std::lock_guard lock(mutex_);
    return store_.size();
}

size_t PrefixStore::total_bytes() const {
    std::lock_guard lock(mutex_);
    size_t total = 0;
    for (const auto& [_, snap] : store_) {
        if (snap) total += snap->byte_size();
    }
    return total;
}

bool PrefixStore::evict_lru() {
    std::lock_guard lock(mutex_);
    if (store_.empty()) return false;

    uint64_t lru_id = 0;
    uint64_t lru_epoch = std::numeric_limits<uint64_t>::max();
    for (const auto& [id, ep] : access_epoch_) {
        if (ep < lru_epoch) {
            lru_epoch = ep;
            lru_id = id;
        }
    }

    store_.erase(lru_id);
    access_epoch_.erase(lru_id);
    return true;
}

}  // namespace kv3d
