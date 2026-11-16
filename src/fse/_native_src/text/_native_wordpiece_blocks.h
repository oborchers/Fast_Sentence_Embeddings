/* 64-byte character-class masks over normalized UTF-8, followed by a resumable
   word scanner. Padding at q+m consists of 64 readable ASCII spaces.
   Prior art: simdjson's arm64 simd8x64<bool>::to_bitmask (weighted pairwise sums).
   No normalization, lookup, validation or input copying happens here. */
#ifndef FSE_NATIVE_WORDPIECE_BLOCKS_H
#define FSE_NATIVE_WORDPIECE_BLOCKS_H

#include "_native_wordpiece.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "WordPiece's native loads require little-endian byte order"
#endif

/* Two diagnostic builds differ only in this constant. The measured winner will
   remove this switch before delivery. */
#ifndef FSE_WP_SCAN_CAPACITY
#define FSE_WP_SCAN_CAPACITY 64
#endif

typedef struct { uint64_t slow, space, punct; } fse_wp_masks;

FSE_INLINE int fse_wp_punct_byte(unsigned b) {
    return (b >= 0x21 && b <= 0x2F) || (b >= 0x3A && b <= 0x40) ||
           (b >= 0x5B && b <= 0x60) || (b >= 0x7B && b <= 0x7E);
}

FSE_INLINE fse_wp_masks fse_wp_masks_byte(const uint8_t *q) {
    fse_wp_masks out = {0, 0, 0};
    for (int i = 0; i < 64; i++) {
        out.slow |= (uint64_t)(q[i] < 0x20 || q[i] > 0x7E) << i;
        out.space |= (uint64_t)(q[i] == 0x20) << i;
        out.punct |= (uint64_t)fse_wp_punct_byte(q[i]) << i;
    }
    return out;
}

/* Exact per-byte unsigned comparisons: the seven-bit addition never carries out
   of a byte. Excluding x's high bits rejects every non-ASCII byte independently. */
FSE_INLINE uint64_t fse_wp_ge8(uint64_t x, unsigned bound) {
    return ((x & ~FSE_WP_HIGH) + FSE_WP_ONES * (128 - bound)) & ~x & FSE_WP_HIGH;
}
FSE_INLINE uint64_t fse_wp_range8(uint64_t x, unsigned lo, unsigned hi) {
    return fse_wp_ge8(x, lo) & ~fse_wp_ge8(x, hi + 1);
}
FSE_INLINE uint64_t fse_wp_pack8(uint64_t x) {
    return (x * 0x0002040810204081ull) >> 56;
}
FSE_INLINE fse_wp_masks fse_wp_masks_swar(const uint8_t *q) {
    fse_wp_masks out = {0, 0, 0};
    for (int i = 0; i < 64; i += 8) {
        uint64_t x;
        memcpy(&x, q + i, 8);
        uint64_t slow = (x | ~fse_wp_ge8(x, 0x20) | fse_wp_ge8(x, 0x7F)) & FSE_WP_HIGH;
        uint64_t space = fse_wp_range8(x, 0x20, 0x20);
        uint64_t punct = fse_wp_range8(x, 0x21, 0x2F) | fse_wp_range8(x, 0x3A, 0x40) |
                         fse_wp_range8(x, 0x5B, 0x60) | fse_wp_range8(x, 0x7B, 0x7E);
        out.slow |= fse_wp_pack8(slow) << i;
        out.space |= fse_wp_pack8(space) << i;
        out.punct |= fse_wp_pack8(punct) << i;
    }
    return out;
}

#if defined(FSE_X86)
FSE_INLINE __m128i fse_wp_range16(__m128i x, int lo, int hi) {
    return _mm_and_si128(_mm_cmpgt_epi8(x, _mm_set1_epi8((char)(lo - 1))),
                         _mm_cmpgt_epi8(_mm_set1_epi8((char)(hi + 1)), x));
}
FSE_INLINE fse_wp_masks fse_wp_masks_simd(const uint8_t *q) {
    fse_wp_masks out = {0, 0, 0};
    for (int i = 0; i < 64; i += 16) {
        __m128i x = _mm_loadu_si128((const __m128i *)(q + i));
        __m128i printable = fse_wp_range16(x, 0x20, 0x7E);
        __m128i punct = _mm_or_si128(
            _mm_or_si128(fse_wp_range16(x, 0x21, 0x2F), fse_wp_range16(x, 0x3A, 0x40)),
            _mm_or_si128(fse_wp_range16(x, 0x5B, 0x60), fse_wp_range16(x, 0x7B, 0x7E)));
        out.slow |= (uint64_t)((unsigned)_mm_movemask_epi8(printable) ^ 0xFFFFu) << i;
        out.space |= (uint64_t)(unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_set1_epi8(0x20))) << i;
        out.punct |= (uint64_t)(unsigned)_mm_movemask_epi8(punct) << i;
    }
    return out;
}
#elif defined(FSE_NEON)
FSE_INLINE uint64_t fse_wp_pack64(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d) {
    const uint8_t weights[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t w = vld1q_u8(weights);
    uint8x16_t ab = vpaddq_u8(vandq_u8(a, w), vandq_u8(b, w));
    uint8x16_t cd = vpaddq_u8(vandq_u8(c, w), vandq_u8(d, w));
    uint8x16_t abcd = vpaddq_u8(ab, cd);
    return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(abcd, abcd)), 0);
}
FSE_INLINE uint8x16_t fse_wp_range16(uint8x16_t x, unsigned lo, unsigned hi) {
    return vandq_u8(vcgeq_u8(x, vdupq_n_u8((uint8_t)lo)), vcleq_u8(x, vdupq_n_u8((uint8_t)hi)));
}
FSE_INLINE fse_wp_masks fse_wp_masks_simd(const uint8_t *q) {
    uint8x16_t slow[4], space[4], punct[4];
    for (int i = 0; i < 4; i++) {
        uint8x16_t x = vld1q_u8(q + 16 * i);
        slow[i] = vmvnq_u8(fse_wp_range16(x, 0x20, 0x7E));
        space[i] = vceqq_u8(x, vdupq_n_u8(0x20));
        punct[i] = vorrq_u8(
            vorrq_u8(fse_wp_range16(x, 0x21, 0x2F), fse_wp_range16(x, 0x3A, 0x40)),
            vorrq_u8(fse_wp_range16(x, 0x5B, 0x60), fse_wp_range16(x, 0x7B, 0x7E)));
    }
    fse_wp_masks out;
    out.slow = fse_wp_pack64(slow[0], slow[1], slow[2], slow[3]);
    out.space = fse_wp_pack64(space[0], space[1], space[2], space[3]);
    out.punct = fse_wp_pack64(punct[0], punct[1], punct[2], punct[3]);
    return out;
}
#endif

static const char *fse_wp_simd_name(void) {
#if defined(FSE_X86)
    return "sse2";
#elif defined(FSE_NEON)
    return "neon";
#else
    return "swar";
#endif
}

/* Variant zero is constant in production; the other variants are test-only. */
FSE_INLINE fse_wp_masks fse_wp_classify64(const uint8_t *q, int variant) {
    if (variant == 1) return fse_wp_masks_byte(q);
    if (variant == 2) return fse_wp_masks_swar(q);
#if defined(FSE_X86) || defined(FSE_NEON)
    return fse_wp_masks_simd(q);
#else
    return fse_wp_masks_swar(q);
#endif
}

FSE_INLINE int fse_wp_ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long bit;
    _BitScanForward64(&bit, x);
    return (int)bit;
#else
    return __builtin_ctzll(x);
#endif
}

typedef struct {
    int64_t b, open, pend_start, pend_end;
    int j, pending, done;
    fse_wp_masks masks;
} fse_wp_scan;

FSE_INLINE void fse_wp_scan_init(fse_wp_scan *st, const uint8_t *q, int variant) {
    st->b = 0; st->j = 0; st->open = -1;
    st->pend_start = st->pend_end = 0;
    st->pending = st->done = 0;
    st->masks = fse_wp_classify64(q, variant);
}

/* One action per iteration, re-cutting masks after every change of j. Every
   emission commits its state before a capacity return, including slow punctuation. */
FSE_INLINE int fse_wp_scan_words(fse_wp_scan *st, const uint8_t *q, int64_t m,
                                 int64_t *starts, int64_t *ends, int capacity, int variant) {
    int count = 0;
    while (!st->done && count < capacity) {
        if (st->pending) {
            starts[count] = st->pend_start; ends[count++] = st->pend_end;
            st->pending = 0;
            continue;
        }
        while (st->j >= 64) {
            st->b += 64; st->j -= 64;
            if (st->b < m) st->masks = fse_wp_classify64(q + st->b, variant);
        }
        if (st->b >= m) {
            if (st->open >= 0) {
                starts[count] = st->open; ends[count++] = m;
                st->open = -1;
            }
            st->done = 1;
            break;
        }
        uint64_t above = UINT64_MAX << st->j;
        uint64_t slow = st->masks.slow & above;
        int cut = slow ? fse_wp_ctz64(slow) : 64;
        uint64_t segment = above & (cut < 64 ? (1ull << cut) - 1 : UINT64_MAX);
        uint64_t w = ~(st->masks.slow | st->masks.space | st->masks.punct) & segment;
        uint64_t p = st->masks.punct & segment;
        if (st->j < cut) {
            if (st->open >= 0 && !(w & (1ull << st->j))) {
                starts[count] = st->open; ends[count++] = st->b + st->j;
                st->open = -1;
                continue;
            }
            if (!(w | p)) { st->j = cut; continue; }
            int s = st->open >= 0 ? st->j : fse_wp_ctz64(w | p);
            if (p & (1ull << s)) {
                starts[count] = st->b + s; ends[count++] = st->b + s + 1;
                st->j = s + 1;
                continue;
            }
            uint64_t x = (~w) >> s;
            int e = x ? s + fse_wp_ctz64(x) : 64;
            int64_t begin = st->open >= 0 ? st->open : st->b + s;
            st->j = e;
            if (e < cut) {
                starts[count] = begin; ends[count++] = st->b + e;
                st->open = -1;
            } else st->open = begin;
            continue;
        }
        int len;
        int64_t pos = st->b + cut;
        uint32_t c = fse_wp_class_at(q, pos, m, &len);
        if (c && st->open >= 0) {
            starts[count] = st->open; ends[count++] = pos;
            st->open = -1;
        } else if (!c && st->open < 0) st->open = pos;
        if (c & FSE_BERT_PUNCT) {
            st->pend_start = pos; st->pend_end = pos + len; st->pending = 1;
        }
        st->j += len;
    }
    return count;
}

FSE_INLINE int fse_wp_scan_next(fse_wp_scan *st, const uint8_t *q, int64_t m,
                                int64_t *start, int64_t *end) {
    return fse_wp_scan_words(st, q, m, start, end, 1, 0);
}
#endif
