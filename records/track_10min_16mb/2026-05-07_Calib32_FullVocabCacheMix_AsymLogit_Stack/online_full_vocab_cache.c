#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ORDERS 8
#define COEFF_COUNT 32

static const uint64_t ROLLING_COEFFS[COEFF_COUNT] = {
    36313ULL,   27191ULL,   51647ULL,   81929ULL,   131071ULL,  196613ULL,
    262147ULL,  393241ULL,  524309ULL,  655373ULL,  786433ULL,  917521ULL,
    1048583ULL, 1179653ULL, 1310729ULL, 1441801ULL, 1572869ULL, 1703941ULL,
    1835017ULL, 1966087ULL, 2097169ULL, 2228243ULL, 2359319ULL, 2490389ULL,
    2621471ULL, 2752549ULL, 2883617ULL, 3014687ULL, 3145757ULL, 3276833ULL,
    3407903ULL, 3538973ULL,
};

static const uint64_t PAIR_MIX = 1000003ULL;
static const uint64_t TABLE_MIX = 0x9e3779b97f4a7c15ULL;

typedef struct {
    uint64_t key;
    uint32_t total;
    uint32_t top_count;
    uint16_t top_tok;
    uint16_t _pad;
} CtxBucket;

typedef struct {
    uint64_t key;
    uint32_t count;
    uint32_t _pad;
} PairBucket;

typedef struct {
    CtxBucket *ctx;
    uint8_t *ctx_used;
    PairBucket *pair;
    uint8_t *pair_used;
    size_t mask;
} HashTables;

typedef struct {
    int depth;
    HashTables tables;
} NgramOrderState;

typedef struct {
    int vocab_size;
    int use_ngram;
    int use_recency;
    int n_orders;
    int max_depth;
    uint16_t *ring;
    int ring_len;
    int ring_head;
    NgramOrderState orders[MAX_ORDERS];

    double alpha;
    double denom_prior;
    double threshold;
    double inv_threshold_span;
    double ngram_max_lambda;
    double ngram_weight_power;

    double arbiter_max_lambda;
    double ngram_scale;
    double recency_scale;

    int rec_window;
    int rec_head;
    int rec_len;
    uint16_t *rec_queue;
    uint32_t *rec_counts;
    uint16_t rec_best_token;
    uint32_t rec_best_count;
    double rec_alpha;
    double rec_denom_prior;
    double rec_threshold;
    double rec_inv_threshold_span;
    double rec_max_lambda;
    double rec_weight_power;
} NativeCache;

void fv_cache_destroy(void *ptr);

static void tables_free(HashTables *t) {
    if (!t) return;
    free(t->ctx);
    free(t->ctx_used);
    free(t->pair);
    free(t->pair_used);
    t->ctx = NULL;
    t->ctx_used = NULL;
    t->pair = NULL;
    t->pair_used = NULL;
    t->mask = 0;
}

static int tables_init(HashTables *t, int bits) {
    if (!t || bits <= 0 || bits > 30) return -1;
    size_t cap = (size_t)1ULL << (size_t)bits;
    t->mask = cap - 1U;
    t->ctx = (CtxBucket *)calloc(cap, sizeof(CtxBucket));
    t->ctx_used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    t->pair = (PairBucket *)calloc(cap, sizeof(PairBucket));
    t->pair_used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!t->ctx || !t->ctx_used || !t->pair || !t->pair_used) {
        tables_free(t);
        return -2;
    }
    return 0;
}

static inline size_t mix_index(uint64_t key, size_t mask) {
    return (size_t)((key * TABLE_MIX) & (uint64_t)mask);
}

static int find_ctx_slot(HashTables *t, uint64_t key, size_t *slot, int *found) {
    size_t idx = mix_index(key, t->mask);
    for (size_t probes = 0; probes <= t->mask; ++probes) {
        if (!t->ctx_used[idx]) {
            *slot = idx;
            *found = 0;
            return 0;
        }
        if (t->ctx[idx].key == key) {
            *slot = idx;
            *found = 1;
            return 0;
        }
        idx = (idx + 1U) & t->mask;
    }
    return -1;
}

static int find_pair_slot(HashTables *t, uint64_t key, size_t *slot, int *found) {
    size_t idx = mix_index(key, t->mask);
    for (size_t probes = 0; probes <= t->mask; ++probes) {
        if (!t->pair_used[idx]) {
            *slot = idx;
            *found = 0;
            return 0;
        }
        if (t->pair[idx].key == key) {
            *slot = idx;
            *found = 1;
            return 0;
        }
        idx = (idx + 1U) & t->mask;
    }
    return -1;
}

static uint32_t pair_get(HashTables *t, uint64_t key) {
    size_t slot = 0;
    int found = 0;
    if (find_pair_slot(t, key, &slot, &found) != 0 || !found) return 0U;
    return t->pair[slot].count;
}

static uint32_t pair_increment(HashTables *t, uint64_t key) {
    size_t slot = 0;
    int found = 0;
    if (find_pair_slot(t, key, &slot, &found) != 0) return 0U;
    if (!found) {
        t->pair_used[slot] = 1U;
        t->pair[slot].key = key;
        t->pair[slot].count = 1U;
        return 1U;
    }
    t->pair[slot].count += 1U;
    return t->pair[slot].count;
}

static int ctx_increment(HashTables *t, uint64_t key, uint16_t tok, uint32_t pair_count) {
    size_t slot = 0;
    int found = 0;
    if (find_ctx_slot(t, key, &slot, &found) != 0) return -1;
    if (!found) {
        t->ctx_used[slot] = 1U;
        t->ctx[slot].key = key;
        t->ctx[slot].total = 1U;
        t->ctx[slot].top_count = pair_count;
        t->ctx[slot].top_tok = tok;
        return 0;
    }
    t->ctx[slot].total += 1U;
    if (pair_count > t->ctx[slot].top_count) {
        t->ctx[slot].top_count = pair_count;
        t->ctx[slot].top_tok = tok;
    }
    return 0;
}

static inline uint64_t pair_key(uint64_t ctx_hash, uint16_t tok) {
    return (ctx_hash * PAIR_MIX) ^ (((uint64_t)tok + 1ULL) * 65537ULL);
}

static void ring_push(NativeCache *st, uint16_t tok) {
    if (st->max_depth <= 0) return;
    if (st->ring_len < st->max_depth) {
        st->ring[st->ring_len++] = tok;
        return;
    }
    st->ring[st->ring_head] = tok;
    st->ring_head = (st->ring_head + 1) % st->max_depth;
}

static int context_hash(const NativeCache *st, int depth, uint64_t *out_hash) {
    if (depth <= 0) {
        *out_hash = 0ULL;
        return 1;
    }
    if (st->ring_len < depth) return 0;
    int start = 0;
    if (st->ring_len < st->max_depth) {
        start = st->ring_len - depth;
    } else {
        start = (st->ring_head + st->max_depth - depth) % st->max_depth;
    }
    uint64_t h = 0ULL;
    for (int j = 0; j < depth; ++j) {
        uint16_t tok = st->ring[(start + j) % st->max_depth];
        h ^= ((uint64_t)tok + 1ULL) * ROLLING_COEFFS[(size_t)j % COEFF_COUNT];
    }
    *out_hash = h;
    return 1;
}

static int ngram_predict_update(
    NativeCache *st,
    uint16_t tok,
    float *out_q,
    float *out_lam,
    float *out_top,
    uint8_t *out_hit,
    uint8_t *out_order
) {
    uint64_t hashes[MAX_ORDERS];
    uint8_t valid[MAX_ORDERS];
    double w_sum = 0.0;
    double q_sum = 0.0;
    double strength_sum = 0.0;
    double local_top = 0.0;
    uint8_t local_hit = 0U;
    uint8_t local_order = 0U;

    *out_q = (float)(1.0 / (double)(st->vocab_size > 0 ? st->vocab_size : 1));
    *out_lam = 0.0f;
    *out_top = 0.0f;
    *out_hit = 0U;
    *out_order = 0U;

    for (int i = 0; i < st->n_orders; ++i) {
        valid[i] = (uint8_t)context_hash(st, st->orders[i].depth, &hashes[i]);
    }

    for (int i = 0; i < st->n_orders; ++i) {
        if (!valid[i]) continue;
        HashTables *tab = &st->orders[i].tables;
        size_t slot = 0;
        int found = 0;
        if (find_ctx_slot(tab, hashes[i], &slot, &found) != 0) return -10;
        if (!found) continue;
        CtxBucket *ctx = &tab->ctx[slot];
        if (ctx->total == 0U) continue;
        double denom = (double)ctx->total + st->denom_prior;
        uint64_t pk = pair_key(hashes[i], tok);
        double qy = ((double)pair_get(tab, pk) + st->alpha) / denom;
        double qt = ((double)ctx->top_count + st->alpha) / denom;
        double rel = (qt - st->threshold) * st->inv_threshold_span;
        if (rel <= 0.0) continue;
        double strength = pow(rel, st->ngram_weight_power);
        w_sum += strength;
        q_sum += strength * qy;
        strength_sum += strength;
        if (qt > local_top) {
            local_top = qt;
            local_hit = (uint8_t)(ctx->top_tok == tok);
            int order = st->orders[i].depth + 1;
            local_order = (uint8_t)(order > 255 ? 255 : order);
        }
    }

    if (w_sum > 0.0) {
        double lam = st->ngram_max_lambda * (strength_sum < 1.0 ? strength_sum : 1.0);
        if (lam > st->ngram_max_lambda) lam = st->ngram_max_lambda;
        *out_q = (float)(q_sum / w_sum);
        *out_lam = (float)lam;
        *out_top = (float)local_top;
        *out_hit = local_hit;
        *out_order = local_order;
    }

    for (int i = 0; i < st->n_orders; ++i) {
        if (!valid[i]) continue;
        HashTables *tab = &st->orders[i].tables;
        uint64_t pk = pair_key(hashes[i], tok);
        uint32_t pc = pair_increment(tab, pk);
        if (pc == 0U) return -11;
        if (ctx_increment(tab, hashes[i], tok, pc) != 0) return -12;
    }
    ring_push(st, tok);
    return 0;
}

static void recency_refresh_best(NativeCache *st) {
    uint16_t best_tok = 0U;
    uint32_t best_count = 0U;
    for (int i = 0; i < st->vocab_size; ++i) {
        if (st->rec_counts[i] > best_count) {
            best_count = st->rec_counts[i];
            best_tok = (uint16_t)i;
        }
    }
    st->rec_best_token = best_tok;
    st->rec_best_count = best_count;
}

static void recency_push(NativeCache *st, uint16_t tok) {
    if (st->rec_window <= 0) return;
    uint8_t refresh = 0U;
    if (st->rec_len >= st->rec_window) {
        uint16_t old = st->rec_queue[st->rec_head];
        if (old < st->vocab_size && st->rec_counts[old] > 0U) {
            st->rec_counts[old] -= 1U;
        }
        if (old == st->rec_best_token) refresh = 1U;
        st->rec_queue[st->rec_head] = tok;
        st->rec_head = (st->rec_head + 1) % st->rec_window;
    } else {
        int idx = (st->rec_head + st->rec_len) % st->rec_window;
        st->rec_queue[idx] = tok;
        st->rec_len += 1;
    }
    if (refresh) recency_refresh_best(st);
    if (tok < st->vocab_size) {
        st->rec_counts[tok] += 1U;
        if (st->rec_counts[tok] > st->rec_best_count) {
            st->rec_best_count = st->rec_counts[tok];
            st->rec_best_token = tok;
        }
    }
}

static void recency_predict_update(
    NativeCache *st,
    uint16_t tok,
    float *out_q,
    float *out_lam,
    float *out_top,
    uint8_t *out_hit,
    uint8_t *out_order
) {
    *out_q = (float)(1.0 / (double)(st->vocab_size > 0 ? st->vocab_size : 1));
    *out_lam = 0.0f;
    *out_top = 0.0f;
    *out_hit = 0U;
    *out_order = 1U;
    if (st->rec_len > 0) {
        double denom = (double)st->rec_len + st->rec_denom_prior;
        double qy = ((double)((tok < st->vocab_size) ? st->rec_counts[tok] : 0U) + st->rec_alpha) / denom;
        double qt = ((double)st->rec_best_count + st->rec_alpha) / denom;
        double rel = (qt - st->rec_threshold) * st->rec_inv_threshold_span;
        if (rel > 0.0) {
            double lam = st->rec_max_lambda * pow(rel, st->rec_weight_power);
            if (lam > st->rec_max_lambda) lam = st->rec_max_lambda;
            *out_q = (float)qy;
            *out_lam = (float)lam;
            *out_top = (float)qt;
            *out_hit = (uint8_t)(st->rec_best_token == tok);
        }
    }
    recency_push(st, tok);
}

void *fv_cache_create(
    const int *orders,
    int n_orders,
    int vocab_size,
    double alpha,
    double threshold,
    double max_lambda,
    double weight_power,
    uint16_t seed_prefix_token,
    int use_ngram,
    int use_recency,
    double arbiter_max_lambda,
    double ngram_scale,
    double recency_scale,
    double rec_alpha,
    double rec_threshold,
    double rec_max_lambda,
    double rec_weight_power,
    int rec_window,
    int table_bits
) {
    NativeCache *st = (NativeCache *)calloc(1, sizeof(NativeCache));
    if (!st) return NULL;
    st->vocab_size = vocab_size > 0 ? vocab_size : 1;
    st->use_ngram = use_ngram;
    st->use_recency = use_recency;
    st->alpha = alpha;
    st->denom_prior = alpha * (double)st->vocab_size;
    st->threshold = threshold;
    st->inv_threshold_span = 1.0 / fmax(1.0 - threshold, 1e-6);
    st->ngram_max_lambda = max_lambda;
    st->ngram_weight_power = weight_power;
    st->arbiter_max_lambda = arbiter_max_lambda;
    st->ngram_scale = ngram_scale;
    st->recency_scale = recency_scale;
    st->rec_alpha = rec_alpha;
    st->rec_denom_prior = rec_alpha * (double)st->vocab_size;
    st->rec_threshold = rec_threshold;
    st->rec_inv_threshold_span = 1.0 / fmax(1.0 - rec_threshold, 1e-6);
    st->rec_max_lambda = rec_max_lambda;
    st->rec_weight_power = rec_weight_power;
    st->rec_window = rec_window > 0 ? rec_window : 0;

    if (st->use_ngram) {
        int prev_depth = -1;
        for (int i = 0; i < n_orders && st->n_orders < MAX_ORDERS; ++i) {
            int depth = orders[i] > 0 ? orders[i] - 1 : 0;
            if (depth == prev_depth) continue;
            prev_depth = depth;
            st->orders[st->n_orders].depth = depth;
            if (depth > st->max_depth) st->max_depth = depth;
            if (tables_init(&st->orders[st->n_orders].tables, table_bits) != 0) {
                fv_cache_destroy(st);
                return NULL;
            }
            st->n_orders += 1;
        }
        if (st->max_depth <= 0) st->max_depth = 1;
        st->ring = (uint16_t *)calloc((size_t)st->max_depth, sizeof(uint16_t));
        if (!st->ring) {
            fv_cache_destroy(st);
            return NULL;
        }
        ring_push(st, seed_prefix_token);
    }

    if (st->use_recency && st->rec_window > 0) {
        st->rec_queue = (uint16_t *)calloc((size_t)st->rec_window, sizeof(uint16_t));
        st->rec_counts = (uint32_t *)calloc((size_t)st->vocab_size, sizeof(uint32_t));
        if (!st->rec_queue || !st->rec_counts) {
            fv_cache_destroy(st);
            return NULL;
        }
        recency_push(st, seed_prefix_token);
    }
    return (void *)st;
}

void fv_cache_destroy(void *ptr) {
    NativeCache *st = (NativeCache *)ptr;
    if (!st) return;
    for (int i = 0; i < st->n_orders; ++i) {
        tables_free(&st->orders[i].tables);
    }
    free(st->ring);
    free(st->rec_queue);
    free(st->rec_counts);
    free(st);
}

int fv_cache_process_chunk(
    void *ptr,
    const uint16_t *tokens,
    int64_t n_tokens,
    float *q_mix_out,
    float *lambda_out,
    float *q_top_out,
    uint8_t *hit_out,
    uint8_t *order_out
) {
    NativeCache *st = (NativeCache *)ptr;
    if (!st || !tokens || n_tokens < 0) return -1;
    const double uniform = 1.0 / (double)(st->vocab_size > 0 ? st->vocab_size : 1);
    for (int64_t i = 0; i < n_tokens; ++i) {
        uint16_t tok = tokens[i];
        double q_acc = 0.0;
        double lam_acc = 0.0;
        double top = 0.0;
        uint8_t hit = 0U;
        uint8_t order = 0U;
        float q = 0.0f, lam = 0.0f, q_top = 0.0f;
        uint8_t h = 0U, od = 0U;

        if (st->use_ngram) {
            int rc = ngram_predict_update(st, tok, &q, &lam, &q_top, &h, &od);
            if (rc != 0) return rc;
            double w = fmin(fmax((double)lam * st->ngram_scale, 0.0), 1.0);
            q_acc += w * (double)q;
            lam_acc += w;
            if ((double)q_top > top) {
                top = (double)q_top;
                hit = h;
                order = od;
            }
        }
        if (st->use_recency) {
            recency_predict_update(st, tok, &q, &lam, &q_top, &h, &od);
            double w = fmin(fmax((double)lam * st->recency_scale, 0.0), 1.0);
            q_acc += w * (double)q;
            lam_acc += w;
            if ((double)q_top > top) {
                top = (double)q_top;
                hit = h;
                order = od;
            }
        }

        double q_mix = uniform;
        double lam_final = lam_acc;
        if (lam_acc > 0.0) {
            q_mix = q_acc / lam_acc;
            if (lam_final > st->arbiter_max_lambda) lam_final = st->arbiter_max_lambda;
        }
        q_mix_out[i] = (float)q_mix;
        lambda_out[i] = (float)lam_final;
        q_top_out[i] = (float)top;
        hit_out[i] = hit;
        order_out[i] = order;
    }
    return 0;
}
