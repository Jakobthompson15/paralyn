"""Owned, one-dimensional contiguous FP32 arrays on the shared native runtime.

Sequence input is explicitly rounded to binary32 and copied. Buffer input must
already be native FP32 and C-contiguous. Operators allocate distinct outputs,
enqueue GPU work and retain its event; to_host/wait/close observe completion.
There is no broadcasting, slicing, borrowing, implicit context or CPU fallback.
"""
from array import array as _host_array
from contextlib import ExitStack
import os
from pathlib import Path

from . import Access, Context, Error, Module, Status, _integer, f32, u32

float32 = "float32"
_MAX_ELEMENTS = (1 << 31) - 1


def _operator_path():
    override = os.environ.get("PARALYN_OPERATORS")
    if override:
        result = Path(override).resolve()
    else:
        package = Path(__file__).resolve().parent
        result = package / "_data" / "operators.prk"
        checkout = package.parents[2]
        if not result.is_file() and (checkout / "CMakeLists.txt").is_file():
            result = checkout / "build" / "operators.prk"
    if not result.is_file():
        raise FileNotFoundError("The validated array operator module is missing; install the "
                                "platform wheel or set PARALYN_OPERATORS to operators.prk")
    return result


class _Operators:
    def __init__(self, context):
        # ExitStack unwinds every previously acquired handle if any load/lookup fails.
        with ExitStack() as acquired:
            self.module = acquired.enter_context(Module.load_file(context, _operator_path()))
            self.add = acquired.enter_context(self.module.kernel("array_add"))
            self.affine = acquired.enter_context(self.module.kernel("array_affine"))
            self.queue = acquired.enter_context(context.queue())
            self._owners = acquired.pop_all()

    def close(self):
        self._owners.close()


def _operators(context):
    context.handle
    resources = getattr(context, "_array_resources", None)
    if resources is None:
        resources = _Operators(context)
        context._array_resources = resources
    return resources


def _shape(shape, size):
    _integer(size, 0, _MAX_ELEMENTS, "array length")
    if shape is None:
        return (size,)
    if not isinstance(shape, (tuple, list)) or len(shape) != 1:
        raise ValueError("only one-dimensional shape=(length,) is supported")
    length = _integer(shape[0], 0, _MAX_ELEMENTS, "shape length")
    if length != size:
        raise ValueError("shape length must equal the number of input FP32 values")
    return (length,)


class Array:
    def __init__(self, *_, **__):
        raise TypeError("Use asarray with an explicit Context, or add/affine to create an Array")

    @classmethod
    def _adopt(cls, context, buffer, shape, event=None, device=None):
        result = cls.__new__(cls)
        result._context, result._buffer, result._event = context, buffer, event
        result._shape, result._closed = shape, False
        result._device = context.device if device is None else device
        return result

    def _open(self):
        if self.closed:
            raise Error(Status.INVALID_HANDLE, "Array", "array is closed")

    @property
    def closed(self):
        return getattr(self, "_closed", True)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._shape[0]

    @property
    def dtype(self):
        return float32

    @property
    def nbytes(self):
        return self.size * 4

    @property
    def device(self):
        return self._device

    def wait(self):
        self._open()
        return self._event.wait() if self._event is not None else None

    def to_host(self):
        self.wait()
        result = _host_array("f")
        result.frombytes(self._buffer.read())
        return result

    def close(self):
        if self.closed:
            return
        self._closed = True
        failure = None
        if self._event is not None:
            try:
                self._event.close()
            except Exception as error:
                failure = error
        try:
            self._buffer.close()
        except Exception as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self, _write=os.write, _exit=os._exit):
        if not self.closed:
            try:
                self.close()
            except Exception as error:
                try:
                    _write(2, f"Paralyn fatal unhandled Array finalizer error: {error}\n".encode())
                finally:
                    _exit(1)


def asarray(values, *, context, dtype=float32, shape=None):
    if not isinstance(context, Context):
        raise TypeError("asarray requires an explicit Context")
    context.handle
    if dtype != float32:
        raise ValueError("only explicit dtype=paralyn.float32 is supported")
    if isinstance(values, (list, tuple)):
        _integer(len(values), 0, _MAX_ELEMENTS, "array length")
        host = _host_array("f", (f32(value).value for value in values))
        data, count = host.tobytes(), len(host)
    else:
        view = memoryview(values)
        if not view.c_contiguous or view.ndim != 1:
            raise ValueError("asarray requires a one-dimensional C-contiguous host buffer")
        if view.itemsize != 4 or view.format not in ("f", "@f", "=f"):
            raise TypeError("buffer input must already contain native-endian FP32 values")
        count = view.nbytes // 4
        _shape(shape, count)
        data = view.tobytes()
    actual_shape = _shape(shape, count)
    buffer = context.buffer(len(data))
    try:
        buffer.write(data)
        return Array._adopt(context, buffer, actual_shape)
    except BaseException:
        buffer.close()
        raise


def _binary(a, b, scalar):
    if not isinstance(a, Array) or not isinstance(b, Array):
        raise TypeError("array operators require two Paralyn Arrays")
    a._open()
    b._open()
    if a._context is not b._context:
        raise Error(Status.CONTEXT_MISMATCH, "array operation", "inputs belong to different contexts")
    if a.shape != b.shape:
        raise ValueError("array shapes must match exactly; broadcasting is unsupported")
    context = a._context
    context.handle
    # Empty results own a valid zero-byte buffer but do not fabricate a GPU event.
    if a.size == 0:
        return Array._adopt(context, context.buffer(0), a.shape, device=a.device)
    resources = _operators(context)
    buffer = context.buffer(a.nbytes)
    event = None
    try:
        with a._buffer.view(access=Access.READ) as av, b._buffer.view(access=Access.READ) as bv, \
                buffer.view(access=Access.WRITE) as out:
            arguments = [av, bv, out, u32(a.size)]
            if scalar is not None:
                arguments.append(scalar)
            event = resources.queue.launch(resources.add if scalar is None else resources.affine,
                                           arguments, grid=((a.size + 255) // 256, 1, 1),
                                           block=(256, 1, 1))
        return Array._adopt(context, buffer, a.shape, event, a.device)
    except BaseException:
        try:
            if event is not None:
                event.close()
        finally:
            buffer.close()
        raise


def add(a, b):
    """Allocate and enqueue elementwise FP32 a+b on the inputs' explicit context."""
    return _binary(a, b, None)


def affine(a, b, scale):
    """Allocate and enqueue FP32 a*scale+b, with contraction disabled by the module policy."""
    return _binary(a, b, f32(scale))
