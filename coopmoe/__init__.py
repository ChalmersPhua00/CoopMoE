"""CoopMoE: two unified-memory boxes serve concurrent requests to an MoE model that is larger
than both boxes' memory. Each box serves its own requests from its own LRU expert cache; a
scheduler sends each request to the box that serves its category, so the two caches hold
different experts, and a box that misses asks its peer before it reads its own NVMe."""
