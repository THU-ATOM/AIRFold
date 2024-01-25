def chunk_generate(lst, chunk_size=64):
    n = len(lst)
    for i in range(n // chunk_size + 1):
        if i * chunk_size < n:
            yield lst[i * chunk_size : i * chunk_size + chunk_size]


# get value from dict by chained keys like a.b.c
def safe_get(d, keys, default=None):
    if not isinstance(keys, list):
        keys = [keys]
    if d is None:
        return default
    for key in keys:
        d = d.get(key)
        if d is None:
            return default
    return d