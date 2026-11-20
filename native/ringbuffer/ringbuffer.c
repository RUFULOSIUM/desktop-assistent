#include "ringbuffer.h"
#include <stdlib.h>
#include <string.h>

int ringbuffer_init(ringbuffer_t *rb, size_t capacity) {
    if (!rb || capacity == 0) {
        return -1;
    }
    rb->buffer = (uint8_t *)malloc(capacity);
    if (!rb->buffer) {
        return -1;
    }
    rb->capacity = capacity;
    rb->head = 0;
    rb->tail = 0;
    return 0;
}

void ringbuffer_free(ringbuffer_t *rb) {
    if (rb && rb->buffer) {
        free(rb->buffer);
        rb->buffer = NULL;
        rb->capacity = 0;
        rb->head = 0;
        rb->tail = 0;
    }
}

size_t ringbuffer_readable(const ringbuffer_t *rb) {
    if (!rb || !rb->buffer) {
        return 0;
    }
    if (rb->head >= rb->tail) {
        return rb->head - rb->tail;
    }
    return rb->capacity - (rb->tail - rb->head);
}

size_t ringbuffer_writable(const ringbuffer_t *rb) {
    if (!rb || !rb->buffer) {
        return 0;
    }
    return rb->capacity - 1 - ringbuffer_readable(rb);
}

size_t ringbuffer_capacity(const ringbuffer_t *rb) {
    if (!rb) {
        return 0;
    }
    return rb->capacity;
}

void ringbuffer_clear(ringbuffer_t *rb) {
    if (rb) {
        rb->head = 0;
        rb->tail = 0;
    }
}

size_t ringbuffer_write(ringbuffer_t *rb, const uint8_t *data, size_t len) {
    if (!rb || !rb->buffer || !data || len == 0) {
        return 0;
    }

    size_t writable = ringbuffer_writable(rb);
    if (len > writable) {
        len = writable;
    }
    if (len == 0) {
        return 0;
    }

    size_t first_chunk = rb->capacity - rb->head;
    if (len <= first_chunk) {
        memcpy(rb->buffer + rb->head, data, len);
        rb->head = (rb->head + len) % rb->capacity;
    } else {
        memcpy(rb->buffer + rb->head, data, first_chunk);
        memcpy(rb->buffer, data + first_chunk, len - first_chunk);
        rb->head = len - first_chunk;
    }
    return len;
}

size_t ringbuffer_read(ringbuffer_t *rb, uint8_t *data, size_t len) {
    if (!rb || !rb->buffer || !data || len == 0) {
        return 0;
    }

    size_t readable = ringbuffer_readable(rb);
    if (len > readable) {
        len = readable;
    }
    if (len == 0) {
        return 0;
    }

    size_t first_chunk = rb->capacity - rb->tail;
    if (len <= first_chunk) {
        memcpy(data, rb->buffer + rb->tail, len);
        rb->tail = (rb->tail + len) % rb->capacity;
    } else {
        memcpy(data, rb->buffer + rb->tail, first_chunk);
        memcpy(data + first_chunk, rb->buffer, len - first_chunk);
        rb->tail = len - first_chunk;
    }
    return len;
}
