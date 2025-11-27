#include <Python.h>
#import <Metal/Metal.h>

static PyObject* is_nax_available(PyObject*, PyObject*) {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) {
    Py_RETURN_FALSE;
  }
  bool has_family9 = false;
  // Apple GPU families 1..15; family 9 corresponds to M3/A17+ class.
  for (int fam = 9; fam <= 15; ++fam) {
    if ([dev supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + fam - 1)]) {
      has_family9 = true;
      break;
    }
  }
  if (has_family9) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef Methods[] = {
    {"is_nax_available", is_nax_available, METH_NOARGS, "Return True if GPU supports Apple family >=9 (NAX class)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "nax_check", nullptr, -1, Methods,
};

PyMODINIT_FUNC PyInit_nax_check(void) { return PyModule_Create(&module); }
