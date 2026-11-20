#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t head;  // write position
    size_t tail;  // read position
    // No locks needed for single-producer/single-consumer
    // head only written by producer, tail only by consumer
} ringbuffer_t;

int ringbuffer_init(ringbuffer_t *rb, size_t capacity);
void ringbuffer_free(ringbuffer_t *rb);
size_t ringbuffer_write(ringbuffer_t *rb, const uint8_t *data, size_t len);
size_t ringbuffer_read(ringbuffer_t *rb, uint8_t *data, size_t len);
size_t ringbuffer_readable(const ringbuffer_t *rb);
size_t ringbuffer_writable(const ringbuffer_t *rb);
size_t ringbuffer_capacity(const ringbuffer_t *rb);
void ringbuffer_clear(ringbuffer_t *rb);

#endif
