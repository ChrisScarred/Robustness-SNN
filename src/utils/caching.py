from dogpile.cache.region import CacheRegion
from dogpile.cache import make_region
from typing import Optional
from functools import lru_cache


@lru_cache
def get_region(name: Optional[str] = None) -> CacheRegion:
    region = None
    if name:
        region = make_region(name=name)
    else:
        region = make_region()
    return region.configure(
        "dogpile.cache.redis",
        arguments={
            "redis_expiration_time": 60 * 60 * 24 * 3,  # 3 days
        },
    )


region = get_region()
