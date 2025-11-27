#include <Python.h>
#import <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <numpy/arrayobject.h>

// Expect two numpy arrays: A (M,K) float16, B (N,K) float16. Returns numpy float32 (M,N).
static PyObject* gemm(PyObject*, PyObject* args) {
  PyObject *a_obj, *b_obj;
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return nullptr;

  PyArrayObject* a_arr = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT16, NPY_ARRAY_ALIGNED);
  PyArrayObject* b_arr = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_FLOAT16, NPY_ARRAY_ALIGNED);
  if (!a_arr || !b_arr) {
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);
    PyErr_SetString(PyExc_TypeError, "Inputs must be float16 numpy arrays");
    return nullptr;
  }
  if (PyArray_NDIM(a_arr) != 2 || PyArray_NDIM(b_arr) != 2) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_ValueError, "Inputs must be 2D");
    return nullptr;
  }
  npy_intp* adims = PyArray_DIMS(a_arr);
  npy_intp* bdims = PyArray_DIMS(b_arr);
  uint32_t M = (uint32_t)adims[0];
  uint32_t K = (uint32_t)adims[1];
  uint32_t N = (uint32_t)bdims[0];
  if ((uint32_t)bdims[1] != K) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_ValueError, "B.shape[1] must match A.shape[1]");
    return nullptr;
  }

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_RuntimeError, "No Metal device");
    return nullptr;
  }

  // Load metallib packaged alongside extension
  NSBundle* bundle = [NSBundle bundleForClass:[NSObject class]];
  NSString* libPath = [[bundle bundlePath] stringByAppendingPathComponent:@"nax_gemm.metallib"];
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithFile:libPath error:&err];
  if (!lib) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_RuntimeError, "Failed to load nax_gemm.metallib");
    return nullptr;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@"nax_gemm"];
  if (!fn) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_RuntimeError, "Kernel nax_gemm not found");
    return nullptr;
  }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create pipeline");
    return nullptr;
  }

  id<MTLCommandQueue> cq = [dev newCommandQueue];
  if (!cq) {
    Py_DECREF(a_arr); Py_DECREF(b_arr);
    PyErr_SetString(PyExc_RuntimeError, "No command queue");
    return nullptr;
  }

  size_t a_bytes = PyArray_NBYTES(a_arr);
  size_t b_bytes = PyArray_NBYTES(b_arr);
  size_t c_bytes = (size_t)M * (size_t)N * sizeof(float);
  id<MTLBuffer> bufA = [dev newBufferWithBytes:PyArray_DATA(a_arr) length:a_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [dev newBufferWithBytes:PyArray_DATA(b_arr) length:b_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [dev newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cb = [cq commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufA offset:0 atIndex:0];
  [enc setBuffer:bufB offset:0 atIndex:1];
  [enc setBuffer:bufC offset:0 atIndex:2];
  uint32_t m=M,n=N,k=K;
  [enc setBytes:&m length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&n length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&k length:sizeof(uint32_t) atIndex:5];

  MTLSize tg = MTLSizeMake(16, 16, 1);
  MTLSize grid = MTLSizeMake((N + tg.width - 1)/tg.width, (M + tg.height - 1)/tg.height, 1);
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  // Build numpy output
  npy_intp dims[2] = {(npy_intp)M, (npy_intp)N};
  PyObject* out_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  memcpy(PyArray_DATA((PyArrayObject*)out_arr), [bufC contents], c_bytes);

  Py_DECREF(a_arr);
  Py_DECREF(b_arr);
  return out_arr;
}

static PyMethodDef Methods[] = {
    {"gemm", gemm, METH_VARARGS, "Naive Metal GEMM (float16 inputs, float32 output)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "nax_gemm", nullptr, -1, Methods,
};

PyMODINIT_FUNC PyInit_nax_gemm(void) {
  import_array();
  return PyModule_Create(&module);
}
