#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "ringbuffer.h"

typedef struct {
    PyObject_HEAD
    ringbuffer_t rb;
} RingBufferObject;

static void RingBuffer_dealloc(RingBufferObject *self) {
    ringbuffer_free(&self->rb);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *RingBuffer_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    RingBufferObject *self = (RingBufferObject *)type->tp_alloc(type, 0);
    if (self) {
        self->rb.buffer = NULL;
        self->rb.capacity = 0;
        self->rb.head = 0;
        self->rb.tail = 0;
    }
    return (PyObject *)self;
}

static int RingBuffer_init(RingBufferObject *self, PyObject *args, PyObject *kwds) {
    Py_ssize_t capacity;
    static char *kwlist[] = {"capacity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &capacity)) {
        return -1;
    }

    if (capacity <= 0) {
        PyErr_SetString(PyExc_ValueError, "capacity must be positive");
        return -1;
    }

    if (ringbuffer_init(&self->rb, (size_t)capacity) != 0) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate ring buffer");
        return -1;
    }
    return 0;
}

static PyObject *RingBuffer_write(RingBufferObject *self, PyObject *args) {
    PyObject *data_obj;
    if (!PyArg_ParseTuple(args, "O", &data_obj)) {
        return NULL;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(data_obj, &view, PyBUF_SIMPLE) < 0) {
        return NULL;
    }

    size_t written = ringbuffer_write(&self->rb, (const uint8_t *)view.buf, view.len);
    PyBuffer_Release(&view);
    return PyLong_FromSize_t(written);
}

static PyObject *RingBuffer_read(RingBufferObject *self, PyObject *args) {
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "n", &size)) {
        return NULL;
    }

    if (size < 0) {
        PyErr_SetString(PyExc_ValueError, "size must be non-negative");
        return NULL;
    }

    size_t readable = ringbuffer_readable(&self->rb);
    if ((size_t)size > readable) {
        size = (Py_ssize_t)readable;
    }

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate read buffer");
        return NULL;
    }

    size_t read = ringbuffer_read(&self->rb, buf, (size_t)size);
    PyObject *result = PyBytes_FromStringAndSize((const char *)buf, read);
    free(buf);
    return result;
}

static PyObject *RingBuffer_readable(RingBufferObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromSize_t(ringbuffer_readable(&self->rb));
}

static PyObject *RingBuffer_writable(RingBufferObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromSize_t(ringbuffer_writable(&self->rb));
}

static PyObject *RingBuffer_capacity(RingBufferObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromSize_t(ringbuffer_capacity(&self->rb));
}

static PyObject *RingBuffer_clear(RingBufferObject *self, PyObject *Py_UNUSED(ignored)) {
    ringbuffer_clear(&self->rb);
    Py_RETURN_NONE;
}

static PyMethodDef RingBuffer_methods[] = {
    {"write", (PyCFunction)RingBuffer_write, METH_VARARGS, "Write data to the buffer"},
    {"read", (PyCFunction)RingBuffer_read, METH_VARARGS, "Read data from the buffer"},
    {"readable", (PyCFunction)RingBuffer_readable, METH_NOARGS, "Return number of readable bytes"},
    {"writable", (PyCFunction)RingBuffer_writable, METH_NOARGS, "Return number of writable bytes"},
    {"capacity", (PyCFunction)RingBuffer_capacity, METH_NOARGS, "Return total capacity"},
    {"clear", (PyCFunction)RingBuffer_clear, METH_NOARGS, "Clear the buffer"},
    {NULL}  /* Sentinel */
};

static PyTypeObject RingBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ringbuffer.RingBuffer",
    .tp_basicsize = sizeof(RingBufferObject),
    .tp_dealloc = (destructor)RingBuffer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Lock-free ring buffer for audio data",
    .tp_methods = RingBuffer_methods,
    .tp_init = (initproc)RingBuffer_init,
    .tp_new = RingBuffer_new,
};

static PyModuleDef ringbuffer_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ringbuffer",
    .m_doc = "Lock-free ring buffer C extension",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_ringbuffer(void) {
    PyObject *m;
    if (PyType_Ready(&RingBufferType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&ringbuffer_module);
    if (!m) {
        return NULL;
    }
    Py_INCREF(&RingBufferType);
    if (PyModule_AddObject(m, "RingBuffer", (PyObject *)&RingBufferType) < 0) {
        Py_DECREF(&RingBufferType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
