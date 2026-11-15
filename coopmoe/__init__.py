"""CoopMoE: two unified-memory boxes serve concurrent requests to an MoE model that is larger
than both boxes' memory. Requests are split at random (half of every category per box), so both boxes need the same
experts; duplicate-first eviction keeps their LRU expert caches different, and a box that misses
asks its peer before it reads its own NVMe."""
