"""Redis-backend caching."""
from functools import lru_cache
from typing import Optional

from dogpile.cache import make_region
from dogpile.cache.region import CacheRegion


@lru_cache
def get_region(name: Optional[str] = None, expiration: int = 60 * 60 * 24 * 3) -> CacheRegion:
    """Obtain the cache region to use for Redis-backend caching.

    Lru-cached to ensure no duplicate regions.

    NOTE: This project also uses lru_cache for caching of data-wise small but frequently called functions. This caching method persists through a single runtime.
    
    Args:
        name (Optional[str], optional): An optional name for this region. Defaults to `None`.
        expiration (int, optional): An optional custom expiration time. The default expiration time is `60 * 60 * 24 * 3`, which indicates 3 days.

    Returns:
        CacheRegion: The CacheRegion element that can be used to annotate cache-able functions.
    """
    region = None
    if name:
        region = make_region(name=name)
    else:
        region = make_region()
    return region.configure(
        "dogpile.cache.redis",
        arguments={
            "redis_expiration_time": expiration
        },
    )


region = get_region()
