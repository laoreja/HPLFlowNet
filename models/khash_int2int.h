#ifndef KHASH_INT2INT_H
#define KHASH_INT2INT_H

#include "khash.h"

KHASH_MAP_INIT_INT64(i2i, khint64_t)

static inline void *khash_int2int_init(void) {
    return kh_init(i2i);
}

static void khash_int2int_destroy(void *_hash) {
    khash_t(i2i) *hash = (khash_t(i2i)*)_hash;
    if (hash) kh_destroy(i2i, hash); 
}

static inline khint64_t khash_int2int_get(void *_hash, khint64_t key, khint64_t default_value) {
    khash_t(i2i) *hash = (khash_t(i2i)*)_hash;
    khint_t k = kh_get(i2i, hash, key);
    if ( k == kh_end(hash) ) return default_value;
    return kh_val(hash, k);
}

static inline int khash_int2int_set(void *_hash, khint64_t key, khint64_t value)
{
    khint_t k;
    int ret;
    khash_t(i2i) *hash = (khash_t(i2i)*)_hash;
    if ( !hash ) return -1;
    k = kh_put(i2i, hash, key, &ret);
    kh_val(hash,k) = value;
    return k;
}

#endif
