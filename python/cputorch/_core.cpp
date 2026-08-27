#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "minitorch/tensor.h"
#include "minitorch/device.h"

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>

using minitorch::Device;
using minitorch::DeviceType;
using minitorch::Tensor;


/* ============================================================
   Python Tensor object
   ============================================================ */

typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensor;


/* ============================================================
   Helpers
   ============================================================ */

static bool PyObject_ToFloatVector(
    PyObject* object,
    std::vector<float>& output
) {
    if (!PySequence_Check(object)) {
        PyErr_SetString(
            PyExc_TypeError,
            "data must be a sequence of numbers"
        );
        return false;
    }

    Py_ssize_t size = PySequence_Size(object);

    if (size < 0) {
        return false;
    }

    output.clear();
    output.reserve(static_cast<std::size_t>(size));

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item =
            PySequence_GetItem(object, i);

        if (!item) {
            return false;
        }

        double value =
            PyFloat_AsDouble(item);

        Py_DECREF(item);

        if (PyErr_Occurred()) {
            PyErr_SetString(
                PyExc_TypeError,
                "data must contain only numbers"
            );
            return false;
        }

        output.push_back(
            static_cast<float>(value)
        );
    }

    return true;
}


static bool PyObject_ToShape(
    PyObject* object,
    std::vector<std::size_t>& output
) {
    if (!PySequence_Check(object)) {
        PyErr_SetString(
            PyExc_TypeError,
            "shape must be a sequence of integers"
        );
        return false;
    }

    Py_ssize_t size =
        PySequence_Size(object);

    if (size < 0) {
        return false;
    }

    output.clear();
    output.reserve(
        static_cast<std::size_t>(size)
    );

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item =
            PySequence_GetItem(object, i);

        if (!item) {
            return false;
        }

        unsigned long long value =
            PyLong_AsUnsignedLongLong(item);

        Py_DECREF(item);

        if (PyErr_Occurred()) {
            PyErr_SetString(
                PyExc_TypeError,
                "shape must contain only integers"
            );
            return false;
        }

        output.push_back(
            static_cast<std::size_t>(value)
        );
    }

    return true;
}


static PyObject* Tensor_ToPython(
    PyTypeObject* type,
    Tensor&& tensor
) {
    PyTensor* object =
        PyObject_New(
            PyTensor,
            type
        );

    if (!object) {
        return nullptr;
    }

    try {
        object->tensor =
            new Tensor(
                std::move(tensor)
            );
    } catch (const std::exception& error) {
        Py_DECREF(object);

        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );

        return nullptr;
    }

    return reinterpret_cast<PyObject*>(
        object
    );
}


/* ============================================================
   Tensor constructor / destructor
   ============================================================ */

static int PyTensor_init(
    PyTensor* self,
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* data_object = nullptr;
    PyObject* shape_object = nullptr;

    int requires_grad = 0;

    static const char* keywords[] = {
        "data",
        "shape",
        "requires_grad",
        nullptr
    };

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "O|Op",
        const_cast<char**>(keywords),
        &data_object,
        &shape_object,
        &requires_grad
    )) {
        return -1;
    }

    std::vector<float> data;
    std::vector<std::size_t> shape;

    if (!PyObject_ToFloatVector(
        data_object,
        data
    )) {
        return -1;
    }

    if (shape_object) {
        if (!PyObject_ToShape(
            shape_object,
            shape
        )) {
            return -1;
        }
    } else {
        shape.push_back(
            data.size()
        );
    }

    try {
        self->tensor =
            new Tensor(
                data,
                shape,
                Device(DeviceType::CPU),
                requires_grad != 0
            );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );

        return -1;
    }

    return 0;
}


static void PyTensor_dealloc(
    PyTensor* self
) {
    delete self->tensor;
    self->tensor = nullptr;

    Py_TYPE(self)->tp_free(
        reinterpret_cast<PyObject*>(self)
    );
}


/* ============================================================
   repr
   ============================================================ */

static PyObject* PyTensor_repr(
    PyTensor* self
) {
    if (!self->tensor) {
        return PyUnicode_FromString(
            "Tensor(<null>)"
        );
    }

    std::string text =
        self->tensor->repr();

    return PyUnicode_FromString(
        text.c_str()
    );
}


/* ============================================================
   data
   ============================================================ */

static PyObject* PyTensor_data(
    PyTensor* self,
    void*
) {
    const auto& data =
        self->tensor->data();

    PyObject* list =
        PyList_New(
            static_cast<Py_ssize_t>(
                data.size()
            )
        );

    if (!list) {
        return nullptr;
    }

    for (
        std::size_t i = 0;
        i < data.size();
        ++i
    ) {
        PyObject* value =
            PyFloat_FromDouble(
                static_cast<double>(
                    data[i]
                )
            );

        if (!value) {
            Py_DECREF(list);
            return nullptr;
        }

        PyList_SET_ITEM(
            list,
            static_cast<Py_ssize_t>(i),
            value
        );
    }

    return list;
}


/* ============================================================
   shape
   ============================================================ */

static PyObject* PyTensor_shape(
    PyTensor* self,
    void*
) {
    const auto& shape =
        self->tensor->shape();

    PyObject* tuple =
        PyTuple_New(
            static_cast<Py_ssize_t>(
                shape.size()
            )
        );

    if (!tuple) {
        return nullptr;
    }

    for (
        std::size_t i = 0;
        i < shape.size();
        ++i
    ) {
        PyObject* value =
            PyLong_FromSize_t(
                shape[i]
            );

        if (!value) {
            Py_DECREF(tuple);
            return nullptr;
        }

        PyTuple_SET_ITEM(
            tuple,
            static_cast<Py_ssize_t>(i),
            value
        );
    }

    return tuple;
}


/* ============================================================
   size
   ============================================================ */

static PyObject* PyTensor_size(
    PyTensor* self,
    void*
) {
    return PyLong_FromSize_t(
        self->tensor->size()
    );
}


/* ============================================================
   ndim
   ============================================================ */

static PyObject* PyTensor_ndim(
    PyTensor* self,
    void*
) {
    return PyLong_FromSize_t(
        self->tensor->ndim()
    );
}


/* ============================================================
   requires_grad
   ============================================================ */

static PyObject* PyTensor_requires_grad(
    PyTensor* self,
    void*
) {
    if (self->tensor->requires_grad()) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}


/* ============================================================
   has_grad
   ============================================================ */

static PyObject* PyTensor_has_grad(
    PyTensor* self,
    void*
) {
    if (self->tensor->has_grad()) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}


/* ============================================================
   add
   ============================================================ */

static PyObject* PyTensor_add(
    PyTensor* self,
    PyObject* args
) {
    PyObject* other_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O",
        &other_object
    )) {
        return nullptr;
    }

    if (!PyObject_TypeCheck(
        other_object,
        Py_TYPE(self)
    )) {
        PyErr_SetString(
            PyExc_TypeError,
            "add expects a Tensor"
        );
        return nullptr;
    }

    PyTensor* other =
        reinterpret_cast<PyTensor*>(
            other_object
        );

    try {
        Tensor result =
            self->tensor->add(
                *other->tensor
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   subtract
   ============================================================ */

static PyObject* PyTensor_subtract(
    PyTensor* self,
    PyObject* args
) {
    PyObject* other_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O",
        &other_object
    )) {
        return nullptr;
    }

    if (!PyObject_TypeCheck(
        other_object,
        Py_TYPE(self)
    )) {
        PyErr_SetString(
            PyExc_TypeError,
            "subtract expects a Tensor"
        );
        return nullptr;
    }

    PyTensor* other =
        reinterpret_cast<PyTensor*>(
            other_object
        );

    try {
        Tensor result =
            self->tensor->subtract(
                *other->tensor
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   multiply
   ============================================================ */

static PyObject* PyTensor_multiply(
    PyTensor* self,
    PyObject* args
) {
    PyObject* other_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O",
        &other_object
    )) {
        return nullptr;
    }

    if (!PyObject_TypeCheck(
        other_object,
        Py_TYPE(self)
    )) {
        PyErr_SetString(
            PyExc_TypeError,
            "multiply expects a Tensor"
        );
        return nullptr;
    }

    PyTensor* other =
        reinterpret_cast<PyTensor*>(
            other_object
        );

    try {
        Tensor result =
            self->tensor->multiply(
                *other->tensor
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   multiply_scalar
   ============================================================ */

static PyObject* PyTensor_multiply_scalar(
    PyTensor* self,
    PyObject* args
) {
    float scalar = 0.0f;

    if (!PyArg_ParseTuple(
        args,
        "f",
        &scalar
    )) {
        return nullptr;
    }

    try {
        Tensor result =
            self->tensor->multiply_scalar(
                scalar
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   matmul
   ============================================================ */

static PyObject* PyTensor_matmul(
    PyTensor* self,
    PyObject* args
) {
    PyObject* other_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O",
        &other_object
    )) {
        return nullptr;
    }

    if (!PyObject_TypeCheck(
        other_object,
        Py_TYPE(self)
    )) {
        PyErr_SetString(
            PyExc_TypeError,
            "matmul expects a Tensor"
        );
        return nullptr;
    }

    PyTensor* other =
        reinterpret_cast<PyTensor*>(
            other_object
        );

    try {
        Tensor result =
            self->tensor->matmul(
                *other->tensor
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   relu
   ============================================================ */

static PyObject* PyTensor_relu(
    PyTensor* self
) {
    try {
        Tensor result =
            self->tensor->relu();

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   sum
   ============================================================ */

static PyObject* PyTensor_sum(
    PyTensor* self
) {
    try {
        Tensor result =
            self->tensor->sum();

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   add_bias_2d
   ============================================================ */

static PyObject* PyTensor_add_bias_2d(
    PyTensor* self,
    PyObject* args
) {
    PyObject* bias_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O",
        &bias_object
    )) {
        return nullptr;
    }

    if (!PyObject_TypeCheck(
        bias_object,
        Py_TYPE(self)
    )) {
        PyErr_SetString(
            PyExc_TypeError,
            "add_bias_2d expects a Tensor"
        );
        return nullptr;
    }

    PyTensor* bias =
        reinterpret_cast<PyTensor*>(
            bias_object
        );

    try {
        Tensor result =
            self->tensor->add_bias_2d(
                *bias->tensor
            );

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   item
   ============================================================ */

static PyObject* PyTensor_item(
    PyTensor* self,
    PyObject* args
) {
    std::size_t index = 0;

    if (!PyArg_ParseTuple(
        args,
        "n",
        &index
    )) {
        return nullptr;
    }

    try {
        return PyFloat_FromDouble(
            static_cast<double>(
                self->tensor->item(index)
            )
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   set_item
   ============================================================ */

static PyObject* PyTensor_set_item(
    PyTensor* self,
    PyObject* args
) {
    std::size_t index = 0;
    float value = 0.0f;

    if (!PyArg_ParseTuple(
        args,
        "nf",
        &index,
        &value
    )) {
        return nullptr;
    }

    try {
        self->tensor->set_item(
            index,
            value
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }

    Py_RETURN_NONE;
}


/* ============================================================
   grad
   ============================================================ */

static PyObject* PyTensor_grad(
    PyTensor* self,
    void*
) {
    try {
        Tensor result =
            self->tensor->grad();

        return Tensor_ToPython(
            Py_TYPE(self),
            std::move(result)
        );
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }
}


/* ============================================================
   zero_grad
   ============================================================ */

static PyObject* PyTensor_zero_grad(
    PyTensor* self
) {
    try {
        self->tensor->zero_grad();
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }

    Py_RETURN_NONE;
}


/* ============================================================
   backward
   ============================================================ */

static PyObject* PyTensor_backward(
    PyTensor* self
) {
    try {
        self->tensor->backward();
    } catch (const std::exception& error) {
        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );
        return nullptr;
    }

    Py_RETURN_NONE;
}


/* ============================================================
   Tensor methods
   ============================================================ */


/* ============================================================
   Native Python indexing
   ============================================================ */

static bool PyIndexToSizeT(
    PyObject* object,
    std::size_t& value
) {
    if (!PyLong_Check(object)) {
        return false;
    }

    long long number =
        PyLong_AsLongLong(object);

    if (PyErr_Occurred()) {
        return false;
    }

    if (number < 0) {
        PyErr_SetString(
            PyExc_IndexError,
            "negative Tensor indices are not supported yet"
        );
        return false;
    }

    value =
        static_cast<std::size_t>(number);

    return true;
}


static PyObject* PyTensor_getitem(
    PyTensor* self,
    PyObject* key
) {
    try {

        const auto& shape =
            self->tensor->shape();

        const std::size_t ndim =
            shape.size();

        std::vector<std::size_t> starts(
            ndim,
            0
        );

        std::vector<std::size_t> stops =
            shape;

        std::vector<std::size_t> steps(
            ndim,
            1
        );

        /*
         * -----------------------------------------------------
         * Integer index
         *
         * t[0]
         * -----------------------------------------------------
         */

        if (PyLong_Check(key)) {

            if (ndim == 0) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "cannot index a scalar Tensor"
                );
                return nullptr;
            }

            std::size_t index = 0;

            if (!PyIndexToSizeT(
                key,
                index
            )) {
                return nullptr;
            }

            if (index >= shape[0]) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "Tensor index out of range"
                );
                return nullptr;
            }

            starts[0] = index;
            stops[0] = index + 1;

            return Tensor_ToPython(
                Py_TYPE(self),
                self->tensor->slice(
                    starts,
                    stops,
                    steps
                )
            );
        }

        /*
         * -----------------------------------------------------
         * Slice
         *
         * t[0:1]
         * -----------------------------------------------------
         */

        if (PySlice_Check(key)) {

            if (ndim == 0) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "cannot slice a scalar Tensor"
                );
                return nullptr;
            }

            Py_ssize_t start;
            Py_ssize_t stop;
            Py_ssize_t step;

            if (PySlice_GetIndices(
                key,
                static_cast<Py_ssize_t>(shape[0]),
                &start,
                &stop,
                &step
            ) < 0) {
                return nullptr;
            }

            if (step <= 0) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Tensor slicing currently requires positive step"
                );
                return nullptr;
            }

            starts[0] =
                static_cast<std::size_t>(start);

            stops[0] =
                static_cast<std::size_t>(stop);

            steps[0] =
                static_cast<std::size_t>(step);

            return Tensor_ToPython(
                Py_TYPE(self),
                self->tensor->slice(
                    starts,
                    stops,
                    steps
                )
            );
        }

        /*
         * -----------------------------------------------------
         * Tuple indexing
         *
         * t[0, 1]
         * t[:, 1]
         * -----------------------------------------------------
         */

        if (PyTuple_Check(key)) {

            const Py_ssize_t count =
                PyTuple_Size(key);

            if (
                count < 1 ||
                static_cast<std::size_t>(count) > ndim
            ) {
                PyErr_SetString(
                    PyExc_IndexError,
                    "invalid number of Tensor indices"
                );
                return nullptr;
            }

            for (
                Py_ssize_t d = 0;
                d < count;
                ++d
            ) {

                PyObject* item =
                    PyTuple_GetItem(
                        key,
                        d
                    );

                /*
                 * Integer
                 */
                if (PyLong_Check(item)) {

                    std::size_t index = 0;

                    if (!PyIndexToSizeT(
                        item,
                        index
                    )) {
                        return nullptr;
                    }

                    const std::size_t dim =
                        static_cast<std::size_t>(d);

                    if (index >= shape[dim]) {
                        PyErr_SetString(
                            PyExc_IndexError,
                            "Tensor index out of range"
                        );
                        return nullptr;
                    }

                    starts[dim] = index;
                    stops[dim] = index + 1;
                    steps[dim] = 1;

                    continue;
                }

                /*
                 * Slice
                 */
                if (PySlice_Check(item)) {

                    const std::size_t dim =
                        static_cast<std::size_t>(d);

                    Py_ssize_t start;
                    Py_ssize_t stop;
                    Py_ssize_t step;

                    if (PySlice_GetIndices(
                        item,
                        static_cast<Py_ssize_t>(
                            shape[dim]
                        ),
                        &start,
                        &stop,
                        &step
                    ) < 0) {
                        return nullptr;
                    }

                    if (step <= 0) {
                        PyErr_SetString(
                            PyExc_ValueError,
                            "Tensor slicing currently requires positive step"
                        );
                        return nullptr;
                    }

                    starts[dim] =
                        static_cast<std::size_t>(start);

                    stops[dim] =
                        static_cast<std::size_t>(stop);

                    steps[dim] =
                        static_cast<std::size_t>(step);

                    continue;
                }

                PyErr_SetString(
                    PyExc_TypeError,
                    "Tensor indices must be integers or slices"
                );

                return nullptr;
            }

            return Tensor_ToPython(
                Py_TYPE(self),
                self->tensor->slice(
                    starts,
                    stops,
                    steps
                )
            );
        }

        PyErr_SetString(
            PyExc_TypeError,
            "Tensor indices must be integers, slices or tuples"
        );

        return nullptr;

    }
    catch (const std::exception& error) {

        PyErr_SetString(
            PyExc_RuntimeError,
            error.what()
        );

        return nullptr;
    }
}
static PyMethodDef PyTensor_methods[] = {

    {
        "add",
        reinterpret_cast<PyCFunction>(
            PyTensor_add
        ),
        METH_VARARGS,
        "Add another Tensor."
    },

    {
        "subtract",
        reinterpret_cast<PyCFunction>(
            PyTensor_subtract
        ),
        METH_VARARGS,
        "Subtract another Tensor."
    },

    {
        "multiply",
        reinterpret_cast<PyCFunction>(
            PyTensor_multiply
        ),
        METH_VARARGS,
        "Element-wise multiplication."
    },

    {
        "multiply_scalar",
        reinterpret_cast<PyCFunction>(
            PyTensor_multiply_scalar
        ),
        METH_VARARGS,
        "Multiply by a scalar."
    },

    {
        "matmul",
        reinterpret_cast<PyCFunction>(
            PyTensor_matmul
        ),
        METH_VARARGS,
        "Matrix multiplication."
    },

    {
        "relu",
        reinterpret_cast<PyCFunction>(
            PyTensor_relu
        ),
        METH_NOARGS,
        "Apply ReLU."
    },

    {
        "sum",
        reinterpret_cast<PyCFunction>(
            PyTensor_sum
        ),
        METH_NOARGS,
        "Sum all elements."
    },

    {
        "add_bias_2d",
        reinterpret_cast<PyCFunction>(
            PyTensor_add_bias_2d
        ),
        METH_VARARGS,
        "Add 2D bias."
    },

    {
        "item",
        reinterpret_cast<PyCFunction>(
            PyTensor_item
        ),
        METH_VARARGS,
        "Get an element."
    },

    {
        "set_item",
        reinterpret_cast<PyCFunction>(
            PyTensor_set_item
        ),
        METH_VARARGS,
        "Set an element."
    },

    {
        "zero_grad",
        reinterpret_cast<PyCFunction>(
            PyTensor_zero_grad
        ),
        METH_NOARGS,
        "Clear gradients."
    },

    {
        "backward",
        reinterpret_cast<PyCFunction>(
            PyTensor_backward
        ),
        METH_NOARGS,
        "Run backward propagation."
    },

    {
        nullptr,
        nullptr,
        0,
        nullptr
    }
};


/* ============================================================
   Tensor properties
   ============================================================ */

static PyObject* PyTensor_dtype(
    PyTensor* self,
    void*
) {
    if (!self || !self->tensor) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Invalid Tensor object."
        );
        return nullptr;
    }

    return PyUnicode_FromString(
        "float32"
    );
}
static PyGetSetDef PyTensor_getset[] = {
    {
        "dtype",
        reinterpret_cast<getter>(
            PyTensor_dtype
        ),
        nullptr,
        "Tensor data type.",
        nullptr
    },


    {
        "data",
        reinterpret_cast<getter>(
            PyTensor_data
        ),
        nullptr,
        "Tensor data.",
        nullptr
    },

    {
        "shape",
        reinterpret_cast<getter>(
            PyTensor_shape
        ),
        nullptr,
        "Tensor shape.",
        nullptr
    },

    {
        "size",
        reinterpret_cast<getter>(
            PyTensor_size
        ),
        nullptr,
        "Number of elements.",
        nullptr
    },

    {
        "ndim",
        reinterpret_cast<getter>(
            PyTensor_ndim
        ),
        nullptr,
        "Number of dimensions.",
        nullptr
    },

    {
        "requires_grad",
        reinterpret_cast<getter>(
            PyTensor_requires_grad
        ),
        nullptr,
        "Whether gradients are required.",
        nullptr
    },

    {
        "has_grad",
        reinterpret_cast<getter>(
            PyTensor_has_grad
        ),
        nullptr,
        "Whether a gradient exists.",
        nullptr
    },

    {
        "grad",
        reinterpret_cast<getter>(
            PyTensor_grad
        ),
        nullptr,
        "Gradient Tensor.",
        nullptr
    },

    {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    }
};


/* ============================================================
   Tensor type
   ============================================================ */


/* ============================================================
   Tensor mapping protocol
   ============================================================ */

static PyMappingMethods PyTensor_mapping = {
    0,
    reinterpret_cast<binaryfunc>(
        PyTensor_getitem
    ),
    0
};
static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};


/* ============================================================
   Module
   ============================================================ */

static PyModuleDef cputorch_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "CPUTorch C++ core.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};


PyMODINIT_FUNC PyInit__core() {

    PyTensorType.tp_name =
        "cputorch._core.Tensor";

    PyTensorType.tp_basicsize =
        sizeof(PyTensor);

    PyTensorType.tp_itemsize =
        0;

    PyTensorType.tp_dealloc =
        reinterpret_cast<destructor>(
            PyTensor_dealloc
        );

    PyTensorType.tp_repr =
        reinterpret_cast<reprfunc>(
            PyTensor_repr
        );

    PyTensorType.tp_flags =
        Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE;

    PyTensorType.tp_as_mapping =
        &PyTensor_mapping;

    PyTensorType.tp_doc =
        "CPUTorch Tensor.";

    PyTensorType.tp_methods =
        PyTensor_methods;

    PyTensorType.tp_getset =
        PyTensor_getset;

    PyTensorType.tp_init =
        reinterpret_cast<initproc>(
            PyTensor_init
        );

    PyTensorType.tp_new =
        PyType_GenericNew;

    if (PyType_Ready(
        &PyTensorType
    ) < 0) {
        return nullptr;
    }

    PyObject* module =
        PyModule_Create(
            &cputorch_module
        );

    if (!module) {
        return nullptr;
    }

    Py_INCREF(
        &PyTensorType
    );

    if (PyModule_AddObject(
        module,
        "Tensor",
        reinterpret_cast<PyObject*>(
            &PyTensorType
        )
    ) < 0) {
        Py_DECREF(
            &PyTensorType
        );
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}



