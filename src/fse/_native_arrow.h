/* Arrow string columns as text sources (ARCHITECTURE.md, "Text sources").

   The structs of the Arrow C Data Interface and the C Stream Interface, as the
   specification asks consumers to copy them (https://arrow.apache.org/docs/format/
   CDataInterface.html, CStreamInterface.html; Apache License 2.0, see NOTICE), and the
   reading of one row of a utf8, large_utf8 or utf8_view chunk.

   fse_arrow_row checks every value it reads before it forms a pointer from it: offsets
   that stay within the data a chunk was imported with, and views whose buffer, offset
   and length stay within the size the producer stated. What it cannot check is the
   producer's obligation: buffers at least as large as their layout implies, unchanged
   while fse reads them. */
#ifndef FSE_NATIVE_ARROW_H
#define FSE_NATIVE_ARROW_H

#include <stdint.h>
#include <string.h>

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callbacks providing stream functionality
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

/* The layouts fse reads. */
#define FSE_ARROW_UTF8 0       /* "u": int32 offsets */
#define FSE_ARROW_LARGE_UTF8 1 /* "U": int64 offsets */
#define FSE_ARROW_UTF8_VIEW 2  /* "vu": 16-byte views and variadic data buffers */

/* A view of at most this many bytes holds its string inline. */
#define FSE_ARROW_INLINE 12

/* One imported chunk. Rows are numbered from 0 to length - 1 within the chunk; first is
   the index of row 0 in the offsets or views (the array offset, or 0 for an own aligned
   copy of them). */
typedef struct {
  int kind;
  int64_t length;
  int64_t first;
  const void* index;           /* int32 or int64 offsets, or the views */
  const unsigned char* data;   /* utf8 and large_utf8: the data buffer */
  int64_t data_len;            /* the bytes up to the last offset of the chunk */
  int64_t n_data;              /* utf8_view: the data buffers and their sizes */
  const unsigned char* const* data_ptrs;
  const int64_t* sizes;
} fse_arrow_chunk;

/* Row i of chunk c as bytes p[0:n]; returns 0, or 1 if its offsets or its view are
   invalid (then p and n are unchanged). */
static inline int fse_arrow_row(const fse_arrow_chunk* c, int64_t i, const unsigned char** p,
                                int64_t* n) {
  int64_t j = c->first + i;
  if (c->kind == FSE_ARROW_UTF8_VIEW) {
    const unsigned char* view = (const unsigned char*)c->index + 16 * j;
    int32_t length, buffer, offset;
    memcpy(&length, view, 4);
    if (length < 0) return 1;
    if (length <= FSE_ARROW_INLINE) {
      *p = view + 4;
      *n = length;
      return 0;
    }
    memcpy(&buffer, view + 8, 4);
    memcpy(&offset, view + 12, 4);
    if (buffer < 0 || buffer >= c->n_data || offset < 0 ||
        (int64_t)offset > c->sizes[buffer] - (int64_t)length)
      return 1;
    *p = c->data_ptrs[buffer] + offset;
    *n = length;
    return 0;
  }
  int64_t start, end;
  if (c->kind == FSE_ARROW_UTF8) {
    const int32_t* offsets = (const int32_t*)c->index;
    start = offsets[j];
    end = offsets[j + 1];
  } else {
    const int64_t* offsets = (const int64_t*)c->index;
    start = offsets[j];
    end = offsets[j + 1];
  }
  if (start < 0 || start > end || end > c->data_len) return 1;
  if (c->data == NULL) { /* only a chunk whose offsets are all 0 has none */
    if (end > 0) return 1;
    *p = c->data;
    *n = 0;
    return 0;
  }
  *p = c->data + start;
  *n = end - start;
  return 0;
}

/* The bytes of row i of a valid utf8_view chunk, without reading its data. */
static inline int64_t fse_arrow_view_length(const fse_arrow_chunk* c, int64_t i) {
  int32_t length;
  memcpy(&length, (const unsigned char*)c->index + 16 * (c->first + i), 4);
  return length;
}

/* The first null among the bits [offset, offset + length) of a validity bitmap (LSB
   order, 1 = valid), as a row from 0, or -1 if there is none. */
static inline int64_t fse_arrow_first_null(const uint8_t* bits, int64_t offset, int64_t length) {
  int64_t i = 0;
  while (i < length) {
    int64_t bit = offset + i;
    if ((bit & 7) == 0 && length - i >= 8) {
      if (bits[bit >> 3] == 0xFF) {
        i += 8;
        continue;
      }
    }
    if (!((bits[bit >> 3] >> (bit & 7)) & 1)) return i;
    i += 1;
  }
  return -1;
}

#endif  // FSE_NATIVE_ARROW_H
