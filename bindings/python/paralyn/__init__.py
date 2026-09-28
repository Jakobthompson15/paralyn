"""Explicit Python ownership wrappers over the Paralyn native C ABI.

This module moves host bytes and launches precompiled, verified kernel artifacts.
It does not compile Python, infer array dtypes, or provide a CPU fallback.
Uploads and module loads accept C-contiguous host buffers only. Strided and
Fortran-only layouts are rejected rather than implicitly reordered.
Use explicit close/context managers to catch release errors. An unexpected
finalizer release error terminates the process with a nonzero exit status.
"""
import ctypes as _c
from dataclasses import dataclass
from enum import IntEnum
import math
import os
from pathlib import Path
import sys

__version__ = "0.0.1"
ABI_VERSION = 1


class Status(IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    INVALID_HANDLE = 2
    CONTEXT_MISMATCH = 3
    OUT_OF_BOUNDS = 4
    UNSUPPORTED = 5
    OUT_OF_MEMORY = 6
    COMPILATION_FAILED = 7
    EXECUTION_FAILED = 8
    DEVICE_UNAVAILABLE = 9
    INTERNAL_ERROR = 10


class Access(IntEnum):
    READ = 1
    WRITE = 2
    READ_WRITE = 3


class Type(IntEnum):
    I32 = 1
    U32 = 2
    F32 = 3
    BUFFER = 4


class Error(RuntimeError):
    def __init__(self, code, operation, message):
        self.code = Status(code)
        self.operation = operation
        self.message = message
        super().__init__(f"{operation}: {message} [{self.code.name}]")


class _ErrorInfo(_c.Structure):
    _fields_ = [("code", _c.c_int), ("operation", _c.c_char * 64),
                ("message", _c.c_char * 1024)]


class _DeviceInfo(_c.Structure):
    _fields_ = [("name", _c.c_char * 256), ("backend", _c.c_char * 32),
                ("os", _c.c_char * 128), ("registry_id", _c.c_uint64),
                ("max_buffer_bytes", _c.c_uint64), ("unified_memory", _c.c_uint32)]


class _Dim3(_c.Structure):
    _fields_ = [("x", _c.c_uint32), ("y", _c.c_uint32), ("z", _c.c_uint32)]


class _Argument(_c.Structure):
    _fields_ = [("type", _c.c_int), ("view", _c.c_uint64), ("i32", _c.c_int32),
                ("u32", _c.c_uint32), ("f32", _c.c_float)]


class _ParameterInfo(_c.Structure):
    _fields_ = [("name", _c.c_char * 4097), ("type", _c.c_int),
                ("is_buffer", _c.c_uint32), ("access", _c.c_int)]


class _EventInfo(_c.Structure):
    _fields_ = [("completed", _c.c_uint32), ("gpu_start_seconds", _c.c_double),
                ("gpu_end_seconds", _c.c_double)]


class _DeviceCapabilities(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("stable_id", _c.c_char * 128), ("backend", _c.c_char * 32),
                ("artifact_formats", _c.c_uint64), ("scalar_types", _c.c_uint64),
                ("max_buffer_bytes", _c.c_uint64), ("max_threadgroup_memory_bytes", _c.c_uint64),
                ("max_block_x", _c.c_uint32), ("max_block_y", _c.c_uint32),
                ("max_block_z", _c.c_uint32), ("max_buffer_bindings", _c.c_uint32),
                ("unified_memory", _c.c_uint32), ("reserved", _c.c_uint32)]


class _EventTiming(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("completed", _c.c_uint32), ("duration_valid", _c.c_uint32),
                ("timestamps_valid", _c.c_uint32), ("clock_domain", _c.c_int),
                ("duration_seconds", _c.c_double), ("start_seconds", _c.c_double),
                ("end_seconds", _c.c_double)]


_QUERY_VERSION_1 = 1


def _query(record_type):
    record = record_type()
    record.struct_size, record.version = _c.sizeof(record_type), _QUERY_VERSION_1
    return record


def _text(value):
    return value.decode("utf-8", errors="replace")


def _integer(value, minimum, maximum, label):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} requires an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _u64(value, label):
    return _integer(value, 0, (1 << 64) - 1, label)


def _cstring(value, label):
    encoded = os.fsencode(value)
    if b"\0" in encoded:
        raise ValueError(f"{label} contains a null byte")
    return encoded


class _Library:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.api = _c.CDLL(self.path)
        h, u32, u64, ptr = _c.c_uint64, _c.c_uint32, _c.c_uint64, _c.POINTER
        signatures = {
            "pr_abi_version": (u32, []),
            "pr_last_error": (_c.c_int, [ptr(_ErrorInfo)]),
            "pr_device_count": (_c.c_int, [ptr(u32)]),
            "pr_device_get": (_c.c_int, [u32, ptr(_DeviceInfo)]),
            "pr_context_create": (_c.c_int, [_c.c_char_p, ptr(h)]),
            "pr_context_device": (_c.c_int, [h, ptr(_DeviceInfo)]),
            "pr_context_synchronize": (_c.c_int, [h]),
            "pr_context_write_evidence": (_c.c_int, [h, _c.c_char_p]),
            "pr_release": (_c.c_int, [h]),
            "pr_buffer_create": (_c.c_int, [h, u64, ptr(h)]),
            "pr_buffer_size": (_c.c_int, [h, ptr(u64)]),
            "pr_view_create": (_c.c_int, [h, u64, u64, u32, _c.c_int, ptr(h)]),
            "pr_buffer_write": (_c.c_int, [h, u64, _c.c_void_p, u64]),
            "pr_buffer_read": (_c.c_int, [h, u64, _c.c_void_p, u64]),
            "pr_module_load": (_c.c_int, [h, _c.c_void_p, u64, ptr(h)]),
            "pr_module_load_file": (_c.c_int, [h, _c.c_char_p, ptr(h)]),
            "pr_module_kernel": (_c.c_int, [h, _c.c_char_p, ptr(h)]),
            "pr_kernel_parameter_count": (_c.c_int, [h, ptr(u32)]),
            "pr_kernel_parameter": (_c.c_int, [h, u32, ptr(_ParameterInfo)]),
            "pr_queue_get": (_c.c_int, [h, ptr(h)]),
            "pr_queue_synchronize": (_c.c_int, [h]),
            "pr_launch": (_c.c_int, [h, h, _Dim3, _Dim3, ptr(_Argument), u32, ptr(h)]),
            "pr_event_wait": (_c.c_int, [h, ptr(_EventInfo)]),
            "pr_event_cancel": (_c.c_int, [h]),
            # Additive versioned queries (ABI 1 records above are unchanged).
            "pr_device_capabilities_get": (_c.c_int, [u32, ptr(_DeviceCapabilities)]),
            "pr_context_capabilities": (_c.c_int, [h, ptr(_DeviceCapabilities)]),
            "pr_event_timing": (_c.c_int, [h, ptr(_EventTiming)]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.api, name)
            function.restype, function.argtypes = result, arguments
        version = self.api.pr_abi_version()
        if version != ABI_VERSION:
            raise RuntimeError(f"Paralyn ABI {version} is incompatible with Python ABI {ABI_VERSION}")

    def check(self, status):
        if status == Status.SUCCESS:
            return
        info = _ErrorInfo()
        retrieval = self.api.pr_last_error(_c.byref(info))
        if retrieval == Status.SUCCESS:
            raise Error(status, _text(info.operation), _text(info.message))
        raise Error(status, "native call", "native error detail retrieval failed")


_libraries = {}
# Platform file name of the native C ABI library (see cli/platform.hpp).
_NATIVE_NAME = {"win32": "paralyn_native.dll", "darwin": "libparalyn_native.dylib"}.get(
    sys.platform, "libparalyn_native.so")


def _library(path=None):
    if path is None:
        path = os.environ.get("PARALYN_LIBRARY")
    if path is None:
        package = Path(__file__).resolve().parent
        bundled = package / "_native" / _NATIVE_NAME
        checkout = package.parents[2]
        if bundled.is_file():
            path = bundled
        elif (checkout / "CMakeLists.txt").is_file():
            path = checkout / "build" / _NATIVE_NAME
        else:
            raise FileNotFoundError("Paralyn's installed native library is missing; reinstall the "
                                    "platform wheel or set PARALYN_LIBRARY explicitly")
    path = str(Path(path).resolve())
    if path not in _libraries:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Build {_NATIVE_NAME} and set PARALYN_LIBRARY "
                                    f"to its path (looked for {path})")
        _libraries[path] = _Library(path)
    return _libraries[path]


@dataclass(frozen=True)
class Device:
    name: str
    backend: str
    os: str
    registry_id: int
    max_buffer_bytes: int
    unified_memory: bool

    @classmethod
    def _from_native(cls, value):
        return cls(_text(value.name), _text(value.backend), _text(value.os), value.registry_id,
                   value.max_buffer_bytes, bool(value.unified_memory))


class ClockDomain(IntEnum):
    UNAVAILABLE = 0
    DURATION_ONLY = 1
    METAL_SYSTEM_MACH = 2


ARTIFACT_VERIFIED_IR = 1 << 0
ARTIFACT_MSL_SOURCE = 1 << 1
ARTIFACT_SPIRV_MSL = 1 << 2


@dataclass(frozen=True)
class DeviceCapabilities:
    """Version-1 capability record; stable_id is backend-qualified and system-local."""
    stable_id: str
    backend: str
    artifact_formats: int
    scalar_types: int
    max_buffer_bytes: int
    max_threadgroup_memory_bytes: int
    max_block: tuple
    max_buffer_bindings: int
    unified_memory: bool

    @classmethod
    def _from_native(cls, value):
        return cls(_text(value.stable_id), _text(value.backend), value.artifact_formats,
                   value.scalar_types, value.max_buffer_bytes, value.max_threadgroup_memory_bytes,
                   (value.max_block_x, value.max_block_y, value.max_block_z),
                   value.max_buffer_bindings, bool(value.unified_memory))


@dataclass(frozen=True)
class EventTiming:
    """Version-1 timing record. Absolute timestamps are None unless the backend marks them valid."""
    completed: bool
    duration_seconds: object
    start_seconds: object
    end_seconds: object
    clock_domain: ClockDomain

    @classmethod
    def _from_native(cls, value):
        duration = value.duration_seconds if value.duration_valid else None
        start = value.start_seconds if value.timestamps_valid else None
        end = value.end_seconds if value.timestamps_valid else None
        return cls(bool(value.completed), duration, start, end, ClockDomain(value.clock_domain))


def device_capabilities(index=0, library=None):
    lib = _library(library)
    value = _query(_DeviceCapabilities)
    lib.check(lib.api.pr_device_capabilities_get(_integer(index, 0, (1 << 32) - 1, "device index"),
                                                 _c.byref(value)))
    return DeviceCapabilities._from_native(value)


def devices(library=None):
    """Available devices: Metal first, then CUDA (see pr_device_get in native.h).

    A position in this tuple is not a context selector: "N" selects Metal
    device N only; CUDA devices are selected with "cuda:K" (K counts CUDA
    entries). registry_id is backend-specific and, for CUDA, not a hardware
    identity.
    """
    lib = _library(library)
    count = _c.c_uint32()
    lib.check(lib.api.pr_device_count(_c.byref(count)))
    result = []
    for index in range(count.value):
        value = _DeviceInfo()
        lib.check(lib.api.pr_device_get(index, _c.byref(value)))
        result.append(Device._from_native(value))
    return tuple(result)


class _Owned:
    @classmethod
    def _adopt(cls, library, handle):
        result = cls.__new__(cls)
        result._lib, result._handle = library, handle
        return result

    @property
    def handle(self):
        if not getattr(self, "_handle", 0):
            raise Error(Status.INVALID_HANDLE, type(self).__name__, "object is closed")
        return self._handle

    @property
    def closed(self):
        return not getattr(self, "_handle", 0)

    def close(self):
        if self.closed:
            return
        handle, self._handle = self._handle, 0
        # Native release consumes a valid handle even when synchronization reports an error.
        self._lib.check(self._lib.api.pr_release(handle))

    def __enter__(self):
        self.handle
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self, _write=os.write, _exit=os._exit):
        if not self.closed:
            try:
                self.close()
            except Exception as error:
                # Python ignores exceptions raised by __del__. Preserve the C++ RAII
                # failure policy instead of allowing an ignored GPU error to exit zero.
                try:
                    _write(2, (f"Paralyn fatal unhandled {type(self).__name__} finalizer error: "
                               f"{error}\n").encode("utf-8", errors="backslashreplace"))
                finally:
                    _exit(1)


class Context(_Owned):
    def __init__(self, selector=None, *, library=None):
        if selector is None:
            selector = os.environ.get("PARALYN_DEVICE", "auto")
        self._lib, self._handle = _library(library), 0
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_context_create(_cstring(selector, "selector"),
                                                       _c.byref(result)))
        self._handle = result.value

    @property
    def device(self):
        value = _DeviceInfo()
        self._lib.check(self._lib.api.pr_context_device(self.handle, _c.byref(value)))
        return Device._from_native(value)

    @property
    def capabilities(self):
        value = _query(_DeviceCapabilities)
        self._lib.check(self._lib.api.pr_context_capabilities(self.handle, _c.byref(value)))
        return DeviceCapabilities._from_native(value)

    def buffer(self, size):
        return Buffer(self, size)

    def queue(self):
        return Queue(self)

    def synchronize(self):
        self._lib.check(self._lib.api.pr_context_synchronize(self.handle))

    def write_evidence(self, directory):
        self._lib.check(self._lib.api.pr_context_write_evidence(
            self.handle, _cstring(directory, "evidence directory")))

    def close(self):
        failure = None
        for name in ("_tensor_resources", "_array_resources"):
            resources = getattr(self, name, None)
            setattr(self, name, None)
            if resources is not None:
                try:
                    resources.close()
                except Exception as error:
                    if failure is None:
                        failure = error
        try:
            super().close()
        except Exception as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure


class Buffer(_Owned):
    def __init__(self, context, size):
        if not isinstance(context, Context):
            raise TypeError("Buffer requires a Context")
        self._lib, self._handle = context._lib, 0
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_buffer_create(context.handle, _u64(size, "buffer size"),
                                                      _c.byref(result)))
        self._handle = result.value

    @property
    def size(self):
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_buffer_size(self.handle, _c.byref(result)))
        return result.value

    def write(self, data, *, offset=0):
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("write requires C-contiguous host bytes; no layout conversion is performed")
        data = view.tobytes()
        storage = _c.create_string_buffer(data, len(data)) if data else None
        self._lib.check(self._lib.api.pr_buffer_write(self.handle, _u64(offset, "write offset"),
                                                     storage, len(data)))

    def read(self, size=None, *, offset=0):
        offset = _u64(offset, "read offset")
        capacity = self.size
        size = capacity - offset if size is None and offset <= capacity else size
        if size is None or offset > capacity:
            raise Error(Status.OUT_OF_BOUNDS, "Buffer.read", "read offset exceeds buffer size")
        size = _u64(size, "read size")
        if size > capacity - offset:
            raise Error(Status.OUT_OF_BOUNDS, "Buffer.read", "read exceeds buffer size")
        storage = _c.create_string_buffer(size) if size else None
        self._lib.check(self._lib.api.pr_buffer_read(self.handle, offset, storage, size))
        return storage.raw if storage is not None else b""

    def view(self, *, offset=0, size=None, alignment=4, access=Access.READ_WRITE):
        return View(self, offset=offset, size=size, alignment=alignment, access=access)


class View(_Owned):
    def __init__(self, buffer, *, offset=0, size=None, alignment=4, access=Access.READ_WRITE):
        if not isinstance(buffer, Buffer):
            raise TypeError("View requires a Buffer")
        self._lib, self._handle = buffer._lib, 0
        offset = _u64(offset, "view offset")
        if size is None:
            size = buffer.size - offset
            if size < 0:
                raise Error(Status.OUT_OF_BOUNDS, "View", "view offset exceeds buffer size")
        size = _u64(size, "view size")
        alignment = _integer(alignment, 0, (1 << 32) - 1, "alignment")
        access = Access(access)
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_view_create(buffer.handle, offset, size, alignment,
                                                    access, _c.byref(result)))
        self._handle = result.value


class Module(_Owned):
    @classmethod
    def load_file(cls, context, path):
        if not isinstance(context, Context):
            raise TypeError("Module.load_file requires a Context")
        result = _c.c_uint64()
        context._lib.check(context._lib.api.pr_module_load_file(
            context.handle, _cstring(path, "module path"), _c.byref(result)))
        return cls._adopt(context._lib, result.value)

    @classmethod
    def load(cls, context, data):
        if not isinstance(context, Context):
            raise TypeError("Module.load requires a Context")
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("Module.load requires C-contiguous artifact bytes; no layout conversion is performed")
        data = view.tobytes()
        storage = _c.create_string_buffer(data, len(data)) if data else None
        result = _c.c_uint64()
        context._lib.check(context._lib.api.pr_module_load(context.handle, storage, len(data),
                                                        _c.byref(result)))
        return cls._adopt(context._lib, result.value)

    def kernel(self, name):
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_module_kernel(self.handle, _cstring(name, "kernel name"),
                                                      _c.byref(result)))
        return Kernel._adopt(self._lib, result.value)


@dataclass(frozen=True)
class Parameter:
    name: str
    type: Type
    is_buffer: bool
    access: Access


class Kernel(_Owned):
    @property
    def parameters(self):
        count = _c.c_uint32()
        self._lib.check(self._lib.api.pr_kernel_parameter_count(self.handle, _c.byref(count)))
        result = []
        for index in range(count.value):
            info = _ParameterInfo()
            self._lib.check(self._lib.api.pr_kernel_parameter(self.handle, index, _c.byref(info)))
            result.append(Parameter(_text(info.name), Type(info.type), bool(info.is_buffer),
                                    Access(info.access)))
        return tuple(result)


@dataclass(frozen=True)
class Scalar:
    type: Type
    value: object

    def __post_init__(self):
        if self.type == Type.I32:
            value = _integer(self.value, -(1 << 31), (1 << 31) - 1, "i32")
        elif self.type == Type.U32:
            value = _integer(self.value, 0, (1 << 32) - 1, "u32")
        elif self.type == Type.F32:
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise TypeError("f32 requires an explicit integer or float value")
            value = float(self.value)
            if math.isfinite(value) and abs(value) > 3.4028234663852886e38:
                raise ValueError("f32 value exceeds the finite FP32 range")
            value = _c.c_float(value).value
        else:
            raise ValueError("Scalar type must be I32, U32 or F32")
        object.__setattr__(self, "value", value)


def i32(value):
    return Scalar(Type.I32, value)


def u32(value):
    return Scalar(Type.U32, value)


def f32(value):
    return Scalar(Type.F32, value)


def _dimensions(value):
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, 1, 1)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError("dimensions require an integer or an (x, y, z) tuple")
    return _Dim3(*(_integer(x, 0, (1 << 32) - 1, "dimension") for x in value))


class Queue(_Owned):
    def __init__(self, context):
        if not isinstance(context, Context):
            raise TypeError("Queue requires a Context")
        self._lib, self._handle = context._lib, 0
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_queue_get(context.handle, _c.byref(result)))
        self._handle = result.value

    def synchronize(self):
        self._lib.check(self._lib.api.pr_queue_synchronize(self.handle))

    def launch(self, kernel, arguments, *, grid, block):
        if not isinstance(kernel, Kernel):
            raise TypeError("launch requires a Kernel")
        if kernel._lib is not self._lib:
            raise ValueError("kernel and queue originate from different native libraries")
        arguments = tuple(arguments)
        count = _integer(len(arguments), 0, (1 << 32) - 1, "argument count")
        native = (_Argument * count)()
        for index, argument in enumerate(arguments):
            if isinstance(argument, View):
                if argument._lib is not self._lib:
                    raise ValueError("view and queue originate from different native libraries")
                native[index].type, native[index].view = Type.BUFFER, argument.handle
            elif isinstance(argument, Scalar):
                native[index].type = argument.type
                setattr(native[index], {Type.I32: "i32", Type.U32: "u32", Type.F32: "f32"}[argument.type],
                        argument.value)
            else:
                raise TypeError("kernel arguments require View or explicit i32/u32/f32 values")
        result = _c.c_uint64()
        self._lib.check(self._lib.api.pr_launch(self.handle, kernel.handle, _dimensions(grid),
                                               _dimensions(block), native, count, _c.byref(result)))
        return Event._adopt(self._lib, result.value)


@dataclass(frozen=True)
class EventInfo:
    completed: bool
    gpu_start_seconds: float
    gpu_end_seconds: float

    @property
    def gpu_duration_seconds(self):
        return self.gpu_end_seconds - self.gpu_start_seconds


class Event(_Owned):
    def wait(self):
        result = _EventInfo()
        self._lib.check(self._lib.api.pr_event_wait(self.handle, _c.byref(result)))
        return EventInfo(bool(result.completed), result.gpu_start_seconds, result.gpu_end_seconds)

    def timing(self):
        """Wait, propagate command failure, and return the versioned timing record."""
        result = _query(_EventTiming)
        self._lib.check(self._lib.api.pr_event_timing(self.handle, _c.byref(result)))
        return EventTiming._from_native(result)

    def cancel(self):
        self._lib.check(self._lib.api.pr_event_cancel(self.handle))


__all__ = ["ABI_VERSION", "Access", "Buffer", "Context", "Device", "Error", "Event",
           "EventInfo", "Kernel", "Module", "Parameter", "Queue", "Scalar", "Status", "Type",
           "View", "devices", "f32", "i32", "u32"]

from .array import Array, add, affine, asarray, float32
__all__ += ["Array", "add", "affine", "asarray", "float32"]
__all__ += ["ARTIFACT_MSL_SOURCE", "ARTIFACT_SPIRV_MSL", "ARTIFACT_VERIFIED_IR", "ClockDomain", "DeviceCapabilities",
            "EventTiming", "device_capabilities"]

from .tensor import (Activation, Tensor, TensorDescriptor, bias_activation_into, bias_add,
                     load_tensor_operators, matmul, matmul_into, relu, tensor,
                     tensor_operators_artifact)
__all__ += ["Activation", "Tensor", "TensorDescriptor", "bias_activation_into", "bias_add",
            "load_tensor_operators", "matmul", "matmul_into", "relu", "tensor",
            "tensor_operators_artifact"]
