try:
    import ast, random, marshal, base64, bz2, zlib, lzma, time, sys, inspect, os, sys, builtins, requests, types, traceback
    import string as _string
    from pystyle import Add,Center,Anime,Colors,Colorate,Write,System
    from sys import platform
    from ast import *
except Exception as e:
    print(e)

antivm = r"""
try:
    _vm = 0
    import uuid, socket, multiprocessing, platform
    mac = ':'.join('%02x' % ((uuid.getnode() >> i) & 0xff) for i in range(0,48,8))[::-1]
    if mac.startswith(('00:05:69','00:0c:29','00:1c:14','00:50:56','08:00:27')):
        _vm += 2
    if any(x in socket.gethostname().lower() for x in ('vmware','vbox','virtual','qemu','xen')):
        _vm += 1
    if multiprocessing.cpu_count() <= 1:
        _vm += 1
    if platform.system() == "Linux":
        try:
            if 'hypervisor' in open('/proc/cpuinfo','r').read().lower():
                _vm += 2
        except:
            pass
    if _vm >= 3:
        open(__file__,'wb').write(b'')
        __import__('sys').exit()
except:
    print("AnhNguyenCoder...")
    __import__("sys").exit()

memory_dump = False
try:
    _os = __import__('os')
    if _os.name == 'nt':
        _ctypes = __import__('ctypes')
        
        kernel32 = _ctypes.windll.kernel32
        if kernel32.IsDebuggerPresent():
            memory_dump = True

        is_remote_debugging = _ctypes.c_int(0)
        kernel32.CheckRemoteDebuggerPresent(-1, _ctypes.byref(is_remote_debugging))
        if is_remote_debugging.value:
            memory_dump = True
except:
    pass

if memory_dump:
    try:
        with open(__file__, "wb") as f:
            f.write(b"")
    except:
        pass
    print("AnhNguyenCoder...")
    __import__('sys').exit()

check_sandbox = False
try:
    _time = AnhNguyenCoder('time')
    _start_time = _time.time()
    _sum = 0
    for i in range(100000):
        _sum += i * i
        if _sum > 1000000000:
            _sum = 0
    
    _end_time = _time.time()
    _elapsed = _end_time - _start_time

    if _elapsed < 0.01 or _elapsed > 10.0:
        check_sandbox = True
except:
    pass

try:
    _socket = __import__('socket')
    _hostname = _socket.gethostname().lower()

    _sandbox_names = ['sandbox', 'malware', 'virus', 'analysis', 'vmware', 'virtualbox', 'vbox', 'qemu', 'xen', 'test', 'lab', 'sample']
    
    for _name in _sandbox_names:
        if _name in _hostname:
            check_sandbox = True
            break
except:
    pass

try:
    _os = __import__('os')
    import multiprocessing
    if multiprocessing.cpu_count() < 2:
        check_sandbox = True
except:
    pass

try:
    _psutil = AnhNguyenCoder('psutil')
    if hasattr(_psutil, 'virtual_memory'):
        _memory = _psutil.virtual_memory()
        if _memory.total < 2 * 1024**3:
            check_sandbox = True
except:
    pass

try:
    _ctypes = __impor__('ctypes')
    
    class _POINT(_ctypes.Structure):
        _fields_ = [("x", _ctypes.c_long), ("y", _ctypes.c_long)]
    
    _pt = _POINT()
    _ctypes.windll.user32.GetCursorPos(_ctypes.byref(_pt))
    
    if _pt.x == 0 and _pt.y == 0:
        _time = AnhNguyenCoder('time')
        _time.sleep(1)
        _ctypes.windll.user32.GetCursorPos(_ctypes.byref(_pt))
        if _pt.x == 0 and _pt.y == 0:
            _check_sandbox = True
except:
    pass

if check_sandbox:
    try:
        with open(__file__, "wb") as f:
            f.write(b"")
    except:
        pass
    print("AnhNguyenCoder...")
    __import__('sys').exit()

try:
    _vm_score = 0
    _dbg_score = 0
    _sb_score  = 0

    def __flag_vm(w=1):
        global _vm_score
        _vm_score += w
    def __flag_dbg(w=1):
        global _dbg_score
        _dbg_score += w
    def __flag_sb(w=1):
        global _sb_score
        _sb_score += w

    try:
        pass
    except:
        pass
    try:
        pass
    except:
        pass
    try:
        pass
    except:
        pass

    _risk = (_vm_score * 3) + (_sb_score * 2) + _dbg_score
    if _risk >= 3:
        raise Exception
except:
    try:
        open(__file__, "wb").write(b"")
    except:
        pass
    print("AnhNguyenCoder...")
    __inport__('sys').exit()"""

antirq = r"""
import sys, os, inspect, subprocess, platform, builtins

def ___sexybeolol___():
    if "PYTHONPATH" in __import__('os').environ or __import__('os').path.exists(__import__('os').path.join(__import__('sys').prefix,"lib","site-packages","sitecustomize.py")):
        print("Anhnguyencoder...")
        raise MemoryError(print)
    if platform.system().lower() != 'windows':
        return
    __codenhucak__ = ['wireshark', 'httptoolkit', 'fiddler', 'httpdebugger', 'charles', 'burp', 'burpsuite', 'mitmproxy', 'mitmdump', 'tcpdump', 'packetsender', 'proxyman', 'tshark', 'httpanalyzer']
    try:
        output = subprocess.check_output('tasklist', shell=True, text=True)
    except Exception:
        return
    output = output.lower()
    for s in __codenhucak__:
        if s.lower() in output:
            raise MemoryError('Anhnguyencoder...')
___sexybeolol___()
__import__('sys').modules.pop('requests', None)

def __anti_hook_url__():
    import inspect
    try:
        from requests.sessions import Session
    except:
        return

    _orig = Session.__dict__.get("request")
    if not callable(_orig):
        return

    def _guard(self, method, url, **kw):
        if Session.request is not _guard:
            raise MemoryError("AnhNguyenCoder...")
        try:
            src = inspect.getsource(Session.request).lower()
            if ("print" in src or "log" in src) and "url" in src:
                raise MemoryError("AnhNguyenCoder...")
        except MemoryError:
            raise
        except:
            pass
        return _orig(self, method, url, **kw)
    Session.request = _guard
__anti_hook_url__()

def hide_url_requests():
    import sys, logging, re, builtins
    try:
        real_print = builtins.print

        def safe_print(*args, **kwargs):
            new_args = []
            for a in args:
                if isinstance(a, str):
                    a = re.sub(r'https?://\S+', '', a)
                new_args.append(a)
            real_print(*new_args, **kwargs)

        setattr(builtins, "print", safe_print)
    except:
        pass

    try:
        from requests.adapters import HTTPAdapter
        original_send = HTTPAdapter.send

        def safe_send(self, request, **kwargs):
            response = original_send(self, request, **kwargs)

            try:
                response.url = ""
                if hasattr(response, "request"):
                    response.request.url = ""
            except:
                pass
            return response
        HTTPAdapter.send = safe_send
    except:
        pass
    try:
        import http.client
        http.client.HTTPConnection.debuglevel = 0
        http.client.HTTPSConnection.debuglevel = 0
    except:
        pass

    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").disabled = True

    sys.settrace(None)
hide_url_requests()

try:
    import sys, os
    if {'sitecustomize','usercustomize'} & sys.modules.keys(): exit(0)
    if any(os.path.isfile(p+os.sep+f) for p in sys.path if p for f in ('sitecustomize.py','usercustomize.py')): exit(0)
    import urllib3
    from urllib3 import PoolManager
    m = getattr(urllib3, '__file__', '').replace('\\\\','/')
    r = PoolManager.request
    if (not m or 'site-packages' not in m
        or not hasattr(r,'__code__')
        or 'urllib3' not in r.__code__.co_filename.replace('\\\\','/')
        or hasattr(r,'__wrapped__')):
        print("Anhnguyencoder...")
        raise MemoryError
except:
    try:
        with open(__file__, "wb") as f:
            f.write(b"")
    except:
        pass
    print("Anhnguyencoder...")
    raise MemoryError
"""

antiglb = r"""
import inspect,sys,types,itertools,importlib,linecache,os,re,dis
from collections import namedtuple, OrderedDict
modulesbyfile = {}
_filesbymodname = {}
_Traceback = namedtuple('_Traceback', 'filename lineno function code_context index')
class Traceback(_Traceback):
    def __new__(cls, __memoryloader__, ___loadrunner__, function, ___occonbo__, ___um___, *, positions=None):
        __anhnguyencoder__ = super().__new__(cls, __memoryloader__, ___loadrunner__, function, ___occonbo__, ___um___)
        __anhnguyencoder__.positions = positions
        return __anhnguyencoder__
    def __repr__(self):
        return 'Traceback(__memoryloader__={!r}, ___loadrunner__={!r}, function={!r}, ___occonbo__={!r}, ___um___={!r}, positions={!r})'.format(self.__memoryloader__, self.___loadrunner__, self.function, self.___occonbo__, self.___um___, self.positions)
_FrameInfo = namedtuple('_FrameInfo', ('frame',) + Traceback._fields)
class FrameInfo(_FrameInfo):
    def __new__(cls, frame, filename, lineno, function, code_context, index, *, positions=None):
        __anhnguyencoder__ = super().__new__(cls, frame, filename, lineno, function, code_context, index)
        __anhnguyencoder__.positions = positions
        return __anhnguyencoder__
    def __repr__(self):
        return 'FrameInfo(frame={!r}, filename={!r}, lineno={!r}, function={!r}, code_context={!r}, index={!r}, positions={!r})'.format(self.frame, self.filename, self.lineno, self.function, self.code_context, self.index, self.positions)
def getabsfile(object, _filename=None):
    if _filename is None:
        _filename = getsourcefile(object) or getfile(object)
    return os.path.normcase(os.path.abspath(_filename))
def getmodule(object, _filename=None):

    if ismodule(object):
        return object
    if hasattr(object, '__module__'):
        return sys.modules.get(object.__module__)
    if _filename is not None and _filename in modulesbyfile:
        return sys.modules.get(modulesbyfile[_filename])
    try:
        file = getabsfile(object, _filename)
    except (TypeError, FileNotFoundError):
        return None
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    for modname, module in sys.modules.copy().items():
        if ismodule(module) and hasattr(module, '__file__'):
            f = module.__file__
            if f == _filesbymodname.get(modname, None):
                continue
            _filesbymodname[modname] = f
            f = getabsfile(module)
            modulesbyfile[f] = modulesbyfile[
                os.path.realpath(f)] = module.__name__
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    main = sys.modules['__main__']
    if not hasattr(object, '__name__'):
        return None
    if hasattr(main, object.__name__):
        mainobject = getattr(main, object.__name__)
        if mainobject is object:
            return main
    builtin = sys.modules['builtins']
    if hasattr(builtin, object.__name__):
        builtinobject = getattr(builtin, object.__name__)
        if builtinobject is object:
            return builtin
def findsource(object):
    file = getsourcefile(object)
    if file:
        linecache.checkcache(file)
    else:
        file = getfile(object)
        if not (file.startswith('<') and file.endswith('>')):
            raise OSError('source code not available')

    module = getmodule(object, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError('could not get source code')

    if ismodule(object):
        return lines, 0

    if isclass(object):
        qualname = object.__qualname__
        source = ''.join(lines)
        tree = ast.parse(source)
        class_finder = _ClassFinder(qualname)
        try:
            class_finder.visit(tree)
        except ClassFoundException as e:
            line_number = e.args[0]
            return lines, line_number
        else:
            raise OSError('could not find class definition')

    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, 'co_firstlineno'):
            raise OSError('could not find function definition')
        lnum = object.co_firstlineno - 1
        pat = re.compile(r'^(\s*def\s)|(\s*async\s+def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)')
        while lnum > 0:
            try:
                line = lines[lnum]
            except IndexError:
                raise OSError('lineno is out of bounds')
            if pat.match(line):
                break
            lnum = lnum - 1
        return lines, lnum
    raise OSError('could not find code object')
def iscode(object):
    return isinstance(object, types.CodeType)
def isframe(object):
    return isinstance(object, types.FrameType)
def ismodule(object):
    return isinstance(object, types.ModuleType)
def isclass(object):
    return isinstance(object, type)
def isfunction(object):
    return isinstance(object, types.FunctionType)
def ismethod(object):
    return isinstance(object, types.MethodType)
def getfile(object):
    if ismodule(object):
        if getattr(object, '__file__', None):
            return object.__file__
        raise TypeError('{!r} is a built-in module'.format(object))
    if isclass(object):
        if hasattr(object, '__module__'):
            module = sys.modules.get(object.__module__)
            if getattr(module, '__file__', None):
                return module.__file__
            if object.__module__ == '__main__':
                raise OSError('source code not available')
        raise TypeError('{!r} is a built-in class'.format(object))
    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        return object.co_filename
    raise TypeError('module, class, method, function, traceback, frame, or ''code object was expected, got {}'.format(type(object).__name__))

def getsourcefile(object):
    filename = getfile(object)
    all_bytecode_suffixes = importlib.machinery.DEBUG_BYTECODE_SUFFIXES[:]
    all_bytecode_suffixes += importlib.machinery.OPTIMIZED_BYTECODE_SUFFIXES[:]
    if any(filename.endswith(s) for s in all_bytecode_suffixes):
        filename = (os.path.splitext(filename)[0] +
                    importlib.machinery.SOURCE_SUFFIXES[0])
    elif any(filename.endswith(s) for s in
                 importlib.machinery.EXTENSION_SUFFIXES):
        return None
    if filename in linecache.cache:
        return filename
    if os.path.exists(filename):
        return filename
    module = getmodule(object, filename)
    if getattr(module, '__loader__', None) is not None:
        return filename
    elif getattr(getattr(module, "__spec__", None), "loader", None) is not None:
        return filename
def istraceback(object):
    return isinstance(object, types.TracebackType)
def _get_code_position(code, instruction_index):
    if instruction_index < 0:
        return (None, None, None, None)
    positions_gen = code.co_positions()
    return next(itertools.islice(positions_gen, instruction_index // 2, None))
def _get_code_position_from_tb(tb):
    code, instruction_index = (tb.tb_frame.f_code, tb.tb_lasti)
    return _get_code_position(code, instruction_index)
def getframeinfo(frame, context=1):
    if istraceback(frame):
        positions = _get_code_position_from_tb(frame)
        lineno = frame.tb_lineno
        frame = frame.tb_frame
    else:
        lineno = frame.f_lineno
        positions = _get_code_position(frame.f_code, frame.f_lasti)
    if positions[0] is None:
        frame, *positions = (frame, lineno, *positions[1:])
    else:
        frame, *positions = (frame, *positions)
    lineno = positions[0]
    if not isframe(frame):
        raise TypeError('{!r} is not a frame or traceback object'.format(frame))
    filename = getsourcefile(frame) or getfile(frame)
    if context > 0:
        start = lineno - 1 - context // 2
        try:
            lines, lnum = findsource(frame)
        except OSError:
            lines = index = None
        else:
            start = max(0, min(start, len(lines) - context))
            lines = lines[start:start + context]
            index = lineno - 1 - start
    else:
        lines = index = None
    return Traceback(filename, lineno, frame.f_code.co_name, lines, index, positions=dis.Positions(*positions))
def __loader__(frame, context=1):
    framelist = []
    while frame:
        traceback_info = getframeinfo(frame, context)
        frameinfo = (frame,) + traceback_info
        framelist.append(FrameInfo(*frameinfo, positions=traceback_info.positions))
        frame = frame.f_back
    return framelist
def stack(context=1):
    return __loader__(sys._getframe(1), context)
def __finally__(__ngauroido__: bytes):
    h = 2166136261
    for b in __ngauroido__:
        h ^= b
        h *= 16777619
        h &= 0xffffffff
    return h

def __ngauroicacem__(code):
    return (code.co_code, code.co_consts, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)

def flatten(__ngauroido__):
    if isinstance(__ngauroido__, (list, tuple)):
        return b''.join(flatten(x) for x in __ngauroido__)
    elif isinstance(__ngauroido__, bytes):
        return __ngauroido__

    elif isinstance(__ngauroido__, str):
        return __ngauroido__.encode('utf-8')
    elif isinstance(__ngauroido__, int):
        return __ngauroido__.to_bytes(8, 'little', signed=True)
    elif __ngauroido__ is None:
        return b'N'
    elif isinstance(__ngauroido__, float):
        import struct
        return struct.pack('<d', __ngauroido__)
    elif isinstance(__ngauroido__, bool):
        return b'T' if __ngauroido__ else b'F'
    elif isinstance(__ngauroido__, type(Ellipsis)):
        return b'E'
    elif isinstance(__ngauroido__, complex):
        import struct
        return struct.pack('<dd', __ngauroido__.real, __ngauroido__.imag)
    elif isinstance(__ngauroido__, type((lambda: 1).__code__)):
        return flatten(__ngauroicacem__(__ngauroido__))
    else:
        return str(__ngauroido__).encode('utf-8')
def __loader1__(code_obj):
    __mmbeo__ = __ngauroicacem__(code_obj)
    __ok__ = flatten(__mmbeo__)
    return __finally__(__ok__)
"""
lolmemaythomlam = """
import os, sys, shutil, zlib, importlib.abc, importlib.util
duoi = ".py__anhnguyencoder___"

def encode_file(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    enc = zlib.compress(data)
    with open(dst, "wb") as f:
        f.write(enc)

def ensure_local_requests():
    try:
        import requests
        src_root = os.path.dirname(requests.__file__)
    except:
        return
    dst_root = os.path.join(os.path.dirname(__file__), "requests")
    if os.path.exists(dst_root):
        return

    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)

        for file in files:
            if file.endswith(".py"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file + duoi)
                encode_file(src_file, dst_file)
            elif not file.endswith((".pyc", ".pyo")):
                shutil.copy2(os.path.join(root, file), os.path.join(dst_dir, file))
class EncLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        with open(self.path, "rb") as f:
            data = zlib.compress(f.read())
        code = compile(data, self.path, "exec")
        exec(code, module.__dict__)

class EncFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("requests"):
            return None

        base = os.path.join(os.path.dirname(__file__), *fullname.split("."))
        file_path = base + duoi
        init_path = os.path.join(base, "__init__.py" + duoi)

        if os.path.isfile(file_path):
            return importlib.util.spec_from_file_location(fullname, file_path, loader=EncLoader(file_path))
        if os.path.isfile(init_path):
            return importlib.util.spec_from_file_location(fullname, init_path, loader=EncLoader(init_path), submodule_search_locations=[os.path.dirname(init_path)])
        return None
ensure_local_requests()
sys.meta_path.insert(0, EncFinder())
p = getattr(__import__('ctypes'), ''.join(['pyt','honapi']))
r = getattr(p, ''.join(['PyMarshal_','ReadObjectFromString']))
e = getattr(p, ''.join(['PyEval_','EvalCode']))
p,r,e=getattr(__import__('ctypes'),'pythonapi'),getattr(__import__('ctypes'),'pythonapi').PyMarshal_ReadObjectFromString,getattr(__import__('ctypes'),'pythonapi').PyEval_EvalCode;[setattr(f,a,v)for f,a,v in[(r,'restype',__import__('ctypes').py_object),(r,'argtypes',[__import__('ctypes').c_char_p,__import__('ctypes').c_long]),(e,'restype',__import__('ctypes').py_object),(e,'argtypes',[__import__('ctypes').py_object]*3)]]
print(' ' * len('Loading...'), end='\\r')
"""
obf_var = r"""
import inspect, sys

def __var_chaos__():
    f = inspect.currentframe().f_back
    fid = id(f)
    globals()[str(fid)] = f.f_lineno ^ fid

    c = 0
    for _ in range(3):
        c += 1
        globals()['_'+('_'*c)] = c << 3

    class _X: pass
    a = _X(); b = _X()
    globals()[str(id(a))] = id(b)
    globals()[str(id(b))] = id(a)

    def _ls_():
        x = 1
        locals()[str(x)] = x << 4
        x = 2
        locals()[str(x)] = x << 5
        return x
    _ls_()
    if sys.gettrace():
        for i in range(1500):
            globals()[str(id(i))] = None

__var_chaos__()
"""

def antibypass():

    def anti(s: str, kkk=69):

        def f(n):
            a, b = (n & 240, n & 15)
            return f'(({a + 10000000000000000000000000}) >>  ({b + 100000000000000000000000000000000000}))' if n > 15 else str(n)
        fx = [f(ord(c) ^ kkk) for c in s]
        mm = ', '.join(fx)
        return f"""((lambda __Anhnguyencoder__: __Anhnguyencoder__(*[__dat__('Biet Dzai Roi',{mm})]))(lambda *__occak__: ((lambda __thknqu__, __Anhnguyencoder__:__Anhnguyencoder__().join([*map(lambda n: __Anhnguyencoder__().format((n ^ 64)), __Anhnguyencoder__)]))(lambda: getattr(''.__class__, '__add__')('__Anhnguyencoder__', ''),lambda: "__CONCAC__"))))"""
    junk_code = []
    for i in range(3000):
        junk_code.append(f'\ndef _junk{i}():\n    x = {random.randint(1000, 9999)}\n    for n in range(50):\n        x ^= (n << {i % 5})\n    return x\n')
    math_noise = []
    for i in range(3000):
        math_noise.append(f'x{i} = ({i} << 2) ^ ({i} * 13)')
    fake_flow = []
    for i in range(3000):
        fake_flow.append(f'\ntry:\n    if ({i} * {i}) % 5 == ({i * i}) % 5:\n        _junk{i % 10}()\nexcept:\n    pass\n')

    def __spam_marshal_runtime__():
        junk_src = 'x=' + str(random.randint(10 ** 50, 10 ** 60))
        junk_ast = ast.parse(junk_src)
        junk_ast = ast.fix_missing_locations(junk_ast)
        blob = marshal.dumps(compile(junk_ast, '<FoNixA>', 'exec'))
        try:
            marshal.loads(blob)
        except:
            pass
        return '0'
    import ast
    def spam_marshal_runtime():
        src = "x='X'*2000000"
        tree = ast.parse(src)
        ast.fix_missing_locations(tree)
        cd = compile(ast.unparse(tree), '<FoNixA>', 'exec')
        blob = marshal.dumps(cd)
        try:
            marshal.loads(blob)
        except:
            pass
        return '0'

    def anti_decompile():
        for _ in range(1000):
            __spam_marshal_runtime__()
            spam_marshal_runtime()
        return '0'

    def mutate_consts():
        import random
        co = mutate_consts.__code__
        junk = bytes((random.randint(0, 255) for _ in range(3000)))
        mutate_consts.__code__ = co.replace(co_consts=co.co_consts + (junk,))
        return '0'
    c = spam_marshal_runtime() + mutate_consts() + anti_decompile() + __spam_marshal_runtime__()

    def _anti():
        def rb():
            return ''.join(random.choices([chr(i) for i in range(44032, 55204) if chr(i).isprintable() and chr(i).isidentifier()], k=11))
        d = rb()
        antipycdc = ''
        for i in range(3000):
            antipycdc += f"__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__('{d}')))))),"
        antipycdc = "try:anhnguyen=[" + antipycdc + c + "]\nexcept:pass"
        text = f"""
{''.join(junk_code)}
def __CTEVCLDZAI__(__chanankdi__):
    return __chanankdi__

try:pass
except:pass
finally:pass
{chr(10).join(math_noise)}
{antipycdc}
{''.join(fake_flow)}
finally:int(2011-2011)
        """
        return f"""
try:
    def __ctevcldz__(__ok__):return "__ANTI-DECOMPILER__"
    {anti("__Anhnguyencoder__")}
except:pass
else:pass
finally:pass
{text}"""

    return _anti()

anti2 = f"""
{antibypass()}
"""

ANTI_MEMORY = r"""
import ctypes, sys, os
try:
    k32 = ctypes.windll.kernel32
    if k32.IsDebuggerPresent():
        raise MemoryError("Anhnguyencoder...")
    flag = ctypes.c_int(0)
    k32.CheckRemoteDebuggerPresent(k32.GetCurrentProcess(), ctypes.byref(flag))
    if flag.value:
        raise MemoryError("Anhnguyencoder...")
    page = 0x40
    mbi = ctypes.create_string_buffer(48)
    addr = id(__builtins__)
    if k32.VirtualQuery(ctypes.c_void_p(addr), mbi, len(mbi)) == 0:
        raise MemoryError("Anhnguyencoder...")
except:
    pass
"""

d_var = r"""
def __dyn_set__(k, v):
    globals()[k] = v

def __dyn_get__(k):
    return globals().get(k)

_k0 = 'x' * 5
_k1 = 'y' * 5

__dyn_set__(_k0, 123456)
__dyn_set__(_k1, __dyn_get__(_k0) ^ 0)
"""

def rb():
    return ''.join(random.choices([chr(i) for i in range(44032, 55204) if chr(i).isprintable() and chr(i).isidentifier()], k=11))

anti = """
import builtins as __b, sys

__real_globals = __b.globals
__real_locals  = __b.locals
__real_vars    = __b.vars
__real_dir     = __b.dir

def __fake_globals():
    g = __real_globals()
    return {k: v for k, v in g.items() if k.startswith('__')}

def __fake_locals():
    l = __real_locals()
    return {k: v for k, v in l.items() if k.startswith('__')}

def __fake_vars(o=None):
    return __fake_locals() if o is None else __real_vars(o)

def __fake_dir(o=None):
    return [n for n in __real_dir(o) if n.startswith('__')] if o else [n for n in __real_dir() if n.startswith('__')]

def __lock_env():
    __b.globals = __fake_globals
    __b.locals  = __fake_locals
    __b.vars    = __fake_vars
    __b.dir     = __fake_dir
"""

jj = """
import sys
sys.setrecursionlimit(999999999) 

def _run(__v1):
    try:
        if isinstance(__v1, bytes):
            __v1 = __v1.decode('utf-8')
        __v2 = __v1[1::2]
        __v3 = bytes.fromhex(__v2)
        return __v3.decode('utf-8')
    except Exception as e:
        return f"[err:{e}]"
"""

def runtime_lock():
    name = rb()
    return f'\nclass {name}(MemoryError): pass\n{jj}\n{anti}'

anti1 = """
try:
    import sys
    if str(__import__('sys').exit) != '<built-in function exit>':
        raise Exception
    if str(print) != '<built-in function print>':
        raise Exception
    if str(exec) != '<built-in function exec>':
        raise Exception
    if str(input) != '<built-in function input>':
        raise Exception
    if str(len) != '<built-in function len>':
        raise Exception
    if str(__import__('marshal').loads) != '<built-in function loads>':
        raise Exception
    with open(__file__, "rb") as f:
        raw = f.read()
except:
    try:
        with open(__file__, "wb") as f:
            f.write(b"")
    except:
        pass
    print(NameError("AnhNguyenCoder..."))
    raise MemoryError

if str(__import__('sys').exit) != '<built-in function exit>':
    raise MemoryError("Anhnguyencoder...")
if str(print) != '<built-in function print>':
    raise MemoryError("Anhnguyencoder...")
if str(exec) != '<built-in function exec>':
    raise MemoryError("Anhnguyencoder...")
if str(input) != '<built-in function input>':
    raise MemoryError("Anhnguyencoder...")
if str(len) != '<built-in function len>':
    raise MemoryError("Anhnguyencoder...")
if str(__import__('marshal').loads) != '<built-in function loads>':
    raise MemoryError("Anhnguyencoder...")

try:
    p3 = __import__("pathlib").Path(__file__).resolve()
except NameError:
    p3 = __import__("pathlib").Path(
        __import__("inspect").getfile(__import__("inspect").currentframe())
    ).resolve()
p1 = __import__("inspect").getfile(__import__("inspect").currentframe())
p2 = __import__("os").path.abspath(p1)
p4 = __import__("pathlib").Path(__import__("__main__").__file__).resolve()
if str(p1) not in str(p2):
    print("AnhNguyenCoder...")
    raise MemoryError
if str(p3) != str(p4):
    print(NameError("AnhNguyenCoder..."))
    raise MemoryError

try:
    if len(open(__file__, encoding='utf-8', errors='ignore').readlines()) != 21:
        raise MemoryError
except:
    try: open(__file__, "wb").write(b"")
    except: pass
    print("AnhNguyenCoder...")
    raise MemoryError

def code_lo_vc():
    try:
        if __OBF__ != ('FonixA'): raise Exception
        if __Author__ != ('Anhnguyencoder'): raise Exception
        if __Tele__ != ('https://t.me/ctevclwar'): raise Exception
        if __In4__ != ('https://www.facebook.com/anhnguyencoder.izumkonata'): raise Exception

        if __CMT__ != {
            "EN": "Việc sử dụng obf này để lạm dụng mục đích xấu, người sở hữu sẽ không chịu trách nghiệm!",
            "VN": "Using this obf for bad purposes, the owner will not be responsible!"
        }:
            raise Exception
    except:
        print(NameError("Anhnguyencoder..."))
        raise MemoryError(print)
code_lo_vc()"""

ver = str(sys.version_info.major)+'.'+str(sys.version_info.minor)
cink = f"""
try:
    if str(__import__('sys').exit) != '<built-in function exit>':
        raise Exception
    if str(print) != '<built-in function print>':
        raise Exception
    if str(exec) != '<built-in function exec>':
        raise Exception
    if str(input) != '<built-in function input>':
        raise Exception
    if str(len) != '<built-in function len>':
        raise Exception
    if str(__import__('marshal').loads) != '<built-in function loads>':
        raise Exception
    with open(__file__, "rb") as f:
        raw = f.read()

    lines = raw.splitlines()
    if len(lines) < 2:
        raise Exception
    if lines[0] != b"#!/bin/python{ver}":
        raise Exception
    if lines[1] not in (b"# -*- coding: utf-8 -*-", b"# coding: utf-8"):
        raise Exception

    scan = lines[2:17]

    for ln in scan:
        s = ln.strip()
        if s.startswith(b"#"):
            raise Exception
        if s.startswith(b"#!"):
            raise Exception
        if s.startswith(b"import "):
            raise Exception
        if s.startswith(b"from ") and b" import " in s:
            raise Exception

    vip = b"\\n".join(scan)

    if b"__OBF__ = ('FonixA')" not in vip:
        raise Exception
    if b"__Author__ = ('Anhnguyencoder')" not in vip:
        raise Exception
    if b"__Tele__" not in vip:
        raise Exception
    if b"__In4__" not in vip:
        raise Exception
    if b"__CMT__" not in vip:
        raise Exception
except:
    try:
        with open(__file__, "wb") as f:
            f.write(b"")
    except: pass
    print(NameError("AnhNguyenCoder..."))
    raise MemoryError
"""

class RenameVars(ast.NodeTransformer):
    def __init__(self):
        self.map = {}
        self.scope_stack = []

    def _new(self):
        return rb()

    def visit_FunctionDef(self, node):
        local_map = {}
        self.scope_stack.append(local_map)

        for arg in node.args.args:
            new = self._new()
            local_map[arg.arg] = new
            arg.arg = new

        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Lambda(self, node):
        local_map = {}
        self.scope_stack.append(local_map)
        for arg in node.args.args:
            new = self._new()
            local_map[arg.arg] = new
            arg.arg = new
        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            if node.id in buitlins:
                return node

            for scope in reversed(self.scope_stack):
                if node.id in scope:
                    node.id = scope[node.id]
                    return node

            if node.id not in self.map:
                self.map[node.id] = self._new()
            node.id = self.map[node.id]

        return node

buitlins = ['__import__', 'abs', 'all', 'any', 'ascii', 'bin', 'breakpoint', 'callable', 'chr', 'compile', 'delattr', 'dir', 'divmod', 'eval', 'exec', 'format', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'id', 'input', 'isinstance', 'issubclass', 'iter', 'aiter', 'len', 'locals', 'max', 'min', 'next', 'anext', 'oct', 'ord', 'pow', 'print', 'repr', 'round', 'setattr', 'sorted', 'sum', 'vars', 'None', 'Ellipsis', 'NotImplemented', 'False', 'True', 'bool', 'memoryview', 'bytearray', 'bytes', 'classmethod', 'complex', 'dict', 'enumerate', 'filter', 'float', 'frozenset', 'property', 'int', 'list', 'map', 'range', 'reversed', 'set', 'slice', 'staticmethod', 'str', 'super', 'tuple', 'type', 'zip', 'print', 'MemoryError', '__dict__']

class hide(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id in buitlins:
            node = Call(func=Name(id='getattr', ctx=Load()), args=[Call(func=Name(id='__import__', ctx=Load()), args=[Constant(value='builtins')], keywords=[]), Constant(value=node.id)], keywords=[])
        return node

import ast
def gen_jcode(code):
    main = rb()
    dzai = rb()
    quadeptrai = rb()
    return [Assign(targets=[Name(id=dzai, ctx=Store())], value=Constant(value=main), lineno=0), Assign(targets=[Name(id=quadeptrai, ctx=Store())], value=Constant(value=True), lineno=0), If(test=BoolOp(op=And(), values=[Compare(left=Name(id=dzai, ctx=Load()), ops=[Eq()], comparators=[Constant(value=main)]), Compare(left=Name(id=quadeptrai, ctx=Load()), ops=[NotEq()], comparators=[Constant(value=True)])]), body=[Expr(value=Lambda(args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=Constant(value='dit me may')))], orelse=[If(test=BoolOp(op=And(), values=[Compare(left=Name(id=dzai, ctx=Load()), ops=[Eq()], comparators=[Constant(value=main)]), Compare(left=Name(id=quadeptrai, ctx=Load()), ops=[NotEq()], comparators=[Constant(value=False)])]), body=[Try(body=[Expr(value=Tuple(elts=[BinOp(left=Constant(value=1), op=Div(), right=Constant(value=0)), BinOp(left=Constant(value=123), op=Div(), right=Constant(value=0)), BinOp(left=Constant(value=12312321312), op=Div(), right=Constant(value=0))], ctx=Load()))], handlers=[ExceptHandler(body=[code])], orelse=[], finalbody=[])], orelse=[If(test=BoolOp(op=Or(), values=[Compare(left=Name(id=dzai, ctx=Load()), ops=[Eq()], comparators=[Constant(value='gay')]), Compare(left=Name(id=quadeptrai, ctx=Load()), ops=[Eq()], comparators=[Constant(value=False)])]), body=[Expr(value=Call(func=Lambda(args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=Call(func=Name(id='print', ctx=Load()), args=[Constant(value='dec dc con cak')], keywords=[])), args=[], keywords=[]))], orelse=[While(test=Constant(value=True), body=[Pass()], orelse=[]), Expr(value=Call(func=Name(id='print', ctx=Load()), args=[Constant(value='ngonbazo')], keywords=[]))])])])]

class junk(ast.NodeTransformer):

    def visit_Module(self, node):
        for i, j in enumerate(node.body):
            if isinstance(j, (ast.FunctionDef, ast.ClassDef)):
                self.visit(j)
            node.body[i] = [gen_jcode(j)]
        return node

    def visit_FunctionDef(self, node):
        for i, j in enumerate(node.body):
            node.body[i] = [gen_jcode(j)]
        return node

def _args(name):
    return ast.arguments(posonlyargs=[], args=[ast.arg(arg=name)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])

def obfstr(s):
    lst = [ord(i) for i in s]
    v = rb()
    lam3 = ast.Lambda(args=_args(rb()), body=ast.Call(func=ast.Attribute(value=ast.Call(ast.Name('exec', ast.Load()), [], []), attr='join', ctx=ast.Load()), args=[ast.GeneratorExp(elt=ast.Call(ast.Name('chr', ast.Load()), [ast.Name(v, ast.Load())], []), generators=[ast.comprehension(target=ast.Name(v, ast.Store()), iter=ast.List([ast.Constant(x) for x in lst], ast.Load()), ifs=[], is_async=0)])], keywords=[]))
    lam2 = ast.Lambda(_args(rb()), ast.Call(lam3, [ast.Constant('AnhNguyenCoder')], []))
    lam1 = ast.Lambda(_args(rb()), ast.Call(lam2, [ast.Constant('AnhNguyenCoder')], []))
    return ast.Call(lam1, [ast.Constant('AnhNguyenCoder')], [])

def obfint(i):
    haha = 2010 - i
    lam3 = ast.Lambda(_args(rb()), ast.Call(ast.Name('exec', ast.Load()), [ast.BinOp(ast.Constant(2010), ast.Sub(), ast.Constant(haha))], []))
    lam2 = ast.Lambda(_args(rb()), ast.Call(lam3, [ast.Constant('AnhNguyenCoder')], []))
    lam1 = ast.Lambda(_args(rb()), ast.Call(lam2, [ast.Constant('AnhNguyenCoder')], []))
    return ast.Call(lam1, [ast.Constant('AnhNguyenCoder')], [])

def joinstr(f):
    if not isinstance(f, ast.JoinedStr):
        return f
    vl = []
    for i in f.values:
        if isinstance(i, ast.Constant):
            vl.append(i)
        elif isinstance(i, ast.FormattedValue):
            value_expr = i.value
            if i.conversion == 115:
                value_expr = Call(func=Name(id='__import__', ctx=Load()), args=[value_expr], keywords=[])
            elif i.conversion == 114:
                value_expr = Call(func=Name(id='repr', ctx=Load()), args=[value_expr], keywords=[])
            elif i.conversion == 97:
                value_expr = Call(func=Name(id='ascii', ctx=Load()), args=[value_expr], keywords=[])
            if i.format_spec:
                if isinstance(i.format_spec, ast.JoinedStr):
                    spec_expr = joinstr(i.format_spec)
                elif isinstance(i.format_spec, ast.Constant):
                    spec_expr = i.format_spec
                elif isinstance(i.format_spec, ast.FormattedValue):
                    spec_parts = []
                    spec_value = i.format_spec.value
                    if i.format_spec.conversion == 115:
                        spec_value = Call(func=Name(id='__import__', ctx=Load()), args=[spec_value], keywords=[])
                    elif i.format_spec.conversion == 114:
                        spec_value = Call(func=Name(id='repr', ctx=Load()), args=[spec_value], keywords=[])
                    elif i.format_spec.conversion == 97:
                        spec_value = Call(func=Name(id='ascii', ctx=Load()), args=[spec_value], keywords=[])
                    spec_expr = spec_value
                else:
                    spec_expr = i.format_spec
                value_expr = Call(func=Name(id='format', ctx=Load()), args=[value_expr, spec_expr], keywords=[])
            elif i.conversion == -1:
                value_expr = Call(func=Name(id='__import__', ctx=Load()), args=[value_expr], keywords=[])
            vl.append(value_expr)
        elif hasattr(i, 'values') and isinstance(i, ast.JoinedStr):
            vl.append(joinstr(i))
        else:
            vl.append(Call(func=Name(id='__import__', ctx=Load()), args=[i], keywords=[]))
    if not vl:
        return Constant(value='')
    if len(vl) == 1 and isinstance(vl[0], ast.Constant):
        return vl[0]
    return Call(func=Attribute(value=Constant(value=''), attr='join', ctx=Load()), args=[Tuple(elts=vl, ctx=Load())], keywords=[])

import ast

class A(ast.NodeTransformer):
    __slots__ = ()
    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        return node
    visit_FunctionDef = visit_Module
    visit_AsyncFunctionDef = visit_Module
    visit_ClassDef = visit_Module

def optimize_ast(code):
    if isinstance(code, ast.AST):
        return Obf().visit(code)
    return code

class cv(ast.NodeTransformer):

    def visit_JoinedStr(self, node):
        node = joinstr(node)
        return node
    
class obf(ast.NodeTransformer):

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node = obfstr(node.value)
        elif isinstance(node.value, int):
            node = obfint(node.value)
        return node

class Flatten(ast.NodeTransformer):

    def visit_FunctionDef(self, node):
        if len(node.body) < 2:
            return node
        state = rb()
        cases = []
        for i, stmt in enumerate(node.body):
            cases.append(ast.If(test=ast.Compare(left=ast.Name(state, ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(i)]), body=[stmt, ast.Assign(targets=[ast.Name(state, ast.Store())], value=ast.Constant(i + 1))], orelse=[]))
        node.body = [ast.Assign(targets=[ast.Name(state, ast.Store())], value=ast.Constant(0)), ast.While(test=ast.Constant(True), body=cases + [ast.Break()], orelse=[])]
        return node

class ConstHide(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 3:
            parts = [node.value[i:i+2] for i in range(0, len(node.value), 2)]
            return ast.Call(
                func=ast.Attribute(value=ast.Constant(''), attr='join', ctx=ast.Load()),
                args=[ast.List(elts=[ast.Constant(p) for p in parts], ctx=ast.Load())],
                keywords=[]
            )
        if isinstance(node.value, int) and node.value > 9:
            a = random.randint(2, 9)
            b = node.value ^ a
            return ast.BinOp(ast.Constant(b), ast.BitXor(), ast.Constant(a))
        return node

class FakeLogic(ast.NodeTransformer):

    def wrap(self, real):
        flag = rb()
        return [ast.Assign(targets=[ast.Name(flag, ast.Store())], value=ast.Constant(True)), ast.If(test=ast.Name(flag, ast.Load()), body=[real], orelse=[])]

    def visit_Module(self, node):
        new_body = []
        for stmt in node.body:
            new_body.extend(self.wrap(stmt))
        node.body = new_body
        return node

    def visit_FunctionDef(self, node):
        new_body = []
        for stmt in node.body:
            new_body.extend(self.wrap(stmt))
        node.body = new_body
        return node

class ASTFormat(ast.NodeTransformer):

    def visit_Expr(self, node):
        return ast.Expr(value=ast.Call(func=ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=node.value), args=[], keywords=[]))

    def visit_Assign(self, node):
        node.value = ast.BinOp(left=ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(0)), op=ast.Sub(), right=ast.Constant(0))
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        node.test = ast.BoolOp(op=ast.And(), values=[ast.Constant(True), node.test])
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.test = ast.BoolOp(op=ast.Or(), values=[node.test, ast.Constant(False)])
        return node

    def visit_Return(self, node):
        if node.value:
            node.value = ast.BinOp(node.value, ast.Add(), ast.Constant(0))
        return node

class hide1(ast.NodeTransformer):
    targets = set(buitlins) | {'exec', 'eval'}

    def _get_builtin(self, name, use_eval=False):
        core = ast.Call(func=ast.Name('getattr', ast.Load()), args=[ast.Call(func=ast.Name('AnhNguyenCoder', ast.Load()), args=[ast.Constant('builtins')], keywords=[]), ast.Constant(name)], keywords=[])
        if use_eval:
            return ast.Call(func=ast.Name('eval', ast.Load()), args=[core], keywords=[])
        return core

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.targets:
            node.func = self._get_builtin(node.func.id, use_eval=node.func.id in {'exec', 'eval'})
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in {'builtins', '__builtins__'} and (node.attr in self.targets):
            return self._get_builtin(node.attr, use_eval=node.attr in {'exec', 'eval'})
        return node

    def visit_Name(self, node):
        if node.id in buitlins:
            node = Call(func=Name(id='getattr', ctx=Load()), args=[Call(func=Name(id='AnhNguyenCoder', ctx=Load()), args=[Constant(value='builtins')], keywords=[]), Constant(value=node.id)], keywords=[])
        return node

class Obf(ast.NodeTransformer):
    def __init__(self):
        self.t_junk = junk()
        self.t_rename = RenameVars()            
        self.t_cv = cv()
        self.t_hide1 = hide1()        
        self.t_hide = hide()        
        self.t_obf = obf()
        self.t_flat = Flatten()
        self.t_const = ConstHide()
        self.t_fake = FakeLogic()
        self.t_rename1 = RenameVars()
        self.t_fmt  = ASTFormat()
        self.t_A = A()

    def visit(self, node):
        node = self.t_junk.visit(node)
        node = self.t_rename.visit(node)          
        node = self.t_cv.visit(node)
        node = self.t_hide1.visit(node)        
        node = self.t_hide.visit(node)        
        node = self.t_obf.visit(node)    
        node = self.t_flat.visit(node)
        node = self.t_const.visit(node)
        node = self.t_fake.visit(node)
        node = self.t_rename1.visit(node)        
        node = self.t_fmt.visit(node)
        node = self.t_A.visit(node)
        
        return node

class AntiSafeVarSpam(ast.NodeTransformer):

    def visit_Module(self, node):
        junk = []
        for i in range(25):
            junk_name = rb()
            junk.append(ast.Assign(targets=[ast.Name(junk_name, ast.Store())], value=ast.BinOp(ast.Constant(i), ast.BitXor(), ast.Constant(123456))))
        node.body = junk + node.body
        return node

class AntiSafeNoise(ast.NodeTransformer):

    def visit_Module(self, node):
        noise = []
        for _ in range(15):
            flag = rb()
            noise.append(ast.Assign(targets=[ast.Name(flag, ast.Store())], value=ast.Constant(True)))
            noise.append(ast.If(test=ast.Name(flag, ast.Load()), body=[ast.Pass()], orelse=[]))
        node.body = noise + node.body
        return node

conconlak = {'__file__', 'filename', 'path', 'p1', 'p2', 'p3', 'p4', 'inspect', 'os', 'sys', 'Path', 'open', 'compile', 'pydc', '__import__', 'exec'}

class Vars(ast.NodeTransformer):

    def __init__(self):
        self.scope_stack = []

    def _new(self):
        return rb()

    def visit_FunctionDef(self, node):
        local_map = {}
        self.scope_stack.append(local_map)
        for arg in node.args.args:
            if arg.arg not in conconlak:
                new = self._new()
                local_map[arg.arg] = new
                arg.arg = new
        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Lambda(self, node):
        local_map = {}
        self.scope_stack.append(local_map)
        for arg in node.args.args:
            if arg.arg not in conconlak:
                new = self._new()
                local_map[arg.arg] = new
                arg.arg = new
        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Name(self, node):
        if node.id in buitlins:
            return node
        if node.id in conconlak:
            return node
        for scope in reversed(self.scope_stack):
            if node.id in scope:
                node.id = scope[node.id]
                return node
        return node

def ast_lol(code: str):
    code = ast.parse(code)

    meo = AntiSafeVarSpam().visit(code)
    code = AntiSafeNoise().visit(meo)
    code = Vars().visit(code)
    
    ast.fix_missing_locations(code)
    return ast.unparse(code)

from ast import *
def lop_gia(code: str):
    import marshal, zlib, base64, random, ast

    co = compile(ast.parse(code), "<FoNixA>", "exec")
    d = marshal.dumps(co)

    k = random.randint(1, 255)
    d = bytes(b ^ k for b in d)
    d = base64.b85encode(zlib.compress(d))

    return "exec(__import__('marshal').loads(bytes(b^%d for b in __import__('zlib').decompress(__import__('base64').b85decode(%r)))))" % (k, d)

import types, random

def anti_pycdc(co: types.CodeType) -> types.CodeType:
    consts = list(co.co_consts)

    deep = ()
    for _ in range(40):
        deep = (deep,)
    consts.append(deep)
    consts.append(bytes(random.getrandbits(8) for _ in range(16384)))

    def __fake__():
        x = 0
        for i in range(32):
            x ^= (i << 2)
        return (lambda y: y ^ x)
    consts.append(__fake__.__code__)
    t = []
    t.append(tuple(t))
    consts.append(tuple(t))
    consts.append(b'\x00PYCDC_POISON\x00')
    return co.replace(co_consts=tuple(consts))

import inspect

string = '0123456789abcdef'
lolrong = list('☠️🗿⭐✦✧✨💫🌠⚡🔥💥☄️🌪❄️🌀🥋')

def anhnguyencoder(s: str) -> str:
    dec = dict(zip(lolrong, string))
    hx = ''.join((dec[x] for x in s.split('|')))
    return bytes.fromhex(hx).decode()

def speed(code):
    import ast
    tree = ast.parse(code) if isinstance(code, str) else code
    enc = dict(zip(string, lolrong))

    class StringBreaker(ast.NodeTransformer):

        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value and ('|' not in node.value) and (not any((e in node.value for e in lolrong))):
                hx = node.value.encode().hex()
                if not all((c in string for c in hx)):
                    return node
                mapped = '|'.join((enc[c] for c in hx))
                return ast.Call(func=ast.Name(id='anhnguyencoder', ctx=ast.Load()), args=[ast.Constant(mapped)], keywords=[])
            return node
    tree = StringBreaker().visit(tree)
    ast.fix_missing_locations(tree)
    return tree

import inspect

string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
lolrong = list('🧸🐲⭐✦✧✨💫🎀👄👁️🔱💢☄️🌪❄️🌀🥋🥊⚔️👊🙌👐🟠🔴🟡🟢🔵🟣⚫⚪👽🤖👺🐢🐒🦍👑💎🔮🍑🍗🍚🍶🏯⛩⛰🛡👑🧙\u200d♂️🤜🤛😡😤🥵🤯🌌🌍🌑☀️🌠')

def anhnguyencoder(s: str) -> str:
    dec = dict(zip(lolrong, string))
    hx = []
    i = 0
    while i < len(s):
        for e, h in dec.items():
            if s.startswith(e, i):
                hx.append(h)
                i += len(e)
                break
        else:
            i += 1
    return bytes.fromhex(''.join(hx)).decode()

def chimto(code):
    import ast
    tree = ast.parse(code) if isinstance(code, str) else code
    enc = dict(zip(string, lolrong))

    class StringBreaker(ast.NodeTransformer):

        def visit_Constant(self, node):
            if not isinstance(node.value, str) or not node.value:
                return node
            if any((e in node.value for e in lolrong)):
                return node
            try:
                raw = node.value.encode()
            except:
                return node
            hx = raw.hex()
            for c in hx:
                if c not in string:
                    return node
            out = []
            for i, c in enumerate(hx):
                e = enc[c]
                if i & 1:
                    e += '\u200d'
                out.append(e)
            mapped = ''.join(out)
            return ast.Call(func=ast.Name(id='anhnguyencoder', ctx=ast.Load()), args=[ast.Constant(mapped)], keywords=[])
    tree = StringBreaker().visit(tree)
    ast.fix_missing_locations(tree)
    return tree

import ast, random, string, marshal

string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
cust = '🧸🐲⭐✦✧✨💫🎀👄👁️🔱💢☄️🌪❄️🌀🥋🥊⚔️👊🙌👐🟠🔴🟡🟢🔵🟣⚫⚪👽🤖👺🐢🐒🦍👑💎🔮🍑🍗🍚🍶🏯⛩⛰🛡👑🧙\u200d♂️🤜🤛😡😤🥵🤯🌌🌍🌑☀️🌠'

memaybeo = dict(zip(string, cust))
vuto = {v: k for k, v in memaybeo.items()}

def phoenixa(s: str) -> str:
    return ''.join(memaybeo[c] for c in s)

def vars(src: str):
    lines = src.splitlines(keepends=True)
    out = []
    for line in lines:
        k = random.randint(20, 240)
        enc = [ord(c) ^ k for c in line[::-1]]
        out.append((enc, k))
    random.shuffle(out)
    return out

def xor(src: str):
    return vars(src)

def lol(b: bytes, width=64) -> str:
    h = b.hex()
    return '\n'.join(h[i:i + width * 2] for i in range(0, len(h), width * 2))

def marshal_load(pairs, mapped_hex):
    enc_pairs = []
    for frag, k in pairs:
        s = ''.join(chr(x) for x in frag)
        s_map = ''.join(memaybeo.get(c, c) for c in s)
        enc_pairs.append((s_map, k))
    return f'''
def var_cai_lol():
    try:
        G = globals(); out = []; chimbeo = {enc_pairs!r}; _dmap = {vuto!r}; i = 0
        while i < len(chimbeo):
            s_map, k = chimbeo[i]; buf = []; p = 0
            while p < len(s_map):
                for x in _dmap:
                    if s_map.startswith(x, p):
                        buf.append(_dmap[x]); p += len(x); break
                else:
                    buf.append(s_map[p]); p += 1
            s = ''.join(buf); gen = (lambda t: [ord(c) for c in t[::-1]])(s); j = 0
            while j < len(gen):
                out.append(chr(gen[j] ^ k))
                j += 1
            i += 1
        vubu = {vuto!r}; _s = {mapped_hex!r}; hx = []; p = 0
        while p < len(_s):
            for k in vubu:
                if _s.startswith(k, p):
                    hx.append(vubu[k])
                    p += len(k)
                    break
        h = ''.join(hx); m = __import__('marshal')
        exec(m.loads(bytes.fromhex(h)), G, G)
    except:pass
var_cai_lol()'''

import ast, random

def moreobf1(src: str) -> str:
    tree = ast.parse(src)

    def rd():
        return '__x0_' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        
    def wrap_stmt(stmt):
        state = rd()
        err = rd()
        assign = ast.Assign(targets=[ast.Name(state, ast.Store())], value=ast.Constant(0))
        try_block = ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name('MemoryError', ast.Load()), args=[ast.Name(state, ast.Load())], keywords=[]))], handlers=[ast.ExceptHandler(type=ast.Name('MemoryError', ast.Load()), name=err, body=[ast.If(test=ast.Compare(left=ast.Subscript(value=ast.Attribute(value=ast.Name(err, ast.Load()), attr='args', ctx=ast.Load()), slice=ast.Constant(0), ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(0)]), body=[stmt], orelse=[])])], orelse=[], finalbody=[])
        return [assign, try_block]
    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.Expr, ast.AugAssign)):
            new_body.extend(wrap_stmt(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = sum([wrap_stmt(n) for n in node.body], [])
            new_body.append(node)
        else:
            new_body.append(node)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

import ast

def _moreobf(tree):
    import random

    def rd():
        return ''.join(random.choices([chr(i) for i in range(12356, 12544) if chr(i).isprintable() and chr(i).isidentifier()], k=11))

    def junk(en, max_value):
        cases = []
        line = max_value + 1
        for i in range(random.randint(1, 5)):
            case_name = '__' + rd()
            case_body = [ast.If(test=ast.Compare(left=ast.Subscript(value=ast.Attribute(value=ast.Name(id=en), attr='args'), slice=ast.Constant(value=0)), ops=[ast.Eq()], comparators=[ast.Constant(value=line)]), body=[ast.Assign(targets=[ast.Name(id=case_name)], value=ast.Constant(value=random.randint(1048575, 281474976710655)), lineno=None)], orelse=[])]
            cases.extend(case_body)
            line += 1
        return cases

    def bl(body):
        var = '__' + rd()
        en = '__' + rd()
        tb = [ast.AugAssign(target=ast.Name(id=var), op=ast.Add(), value=ast.Constant(value=1)), ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='MemoryError'), args=[ast.Name(id=var)], keywords=[]))], handlers=[ast.ExceptHandler(type=ast.Name(id='MemoryError'), name=en, body=[])], orelse=[], finalbody=[])]
        for i in body:
            tb[1].handlers[0].body.append(ast.If(test=ast.Compare(left=ast.Subscript(value=ast.Attribute(value=ast.Name(id=en), attr='args'), slice=ast.Constant(value=0)), ops=[ast.Eq()], comparators=[ast.Constant(value=1)]), body=[i], orelse=[]))
        tb[1].handlers[0].body.extend(junk(en, len(body) + 1))
        node = ast.Assign(targets=[ast.Name(id=var)], value=ast.Constant(value=0), lineno=None)
        return [node] + tb

    def _bl(node):
        olb = node.body
        var = '__' + rd()
        en = '__' + rd()
        tb = [ast.AugAssign(target=ast.Name(id=var), op=ast.Add(), value=ast.Constant(value=1)), ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='MemoryError'), args=[ast.Name(id=var)], keywords=[]))], handlers=[ast.ExceptHandler(type=ast.Name(id='MemoryError'), name=en, body=[])], orelse=[], finalbody=[])]
        for i in olb:
            tb[1].handlers[0].body.append(ast.If(test=ast.Compare(left=ast.Subscript(value=ast.Attribute(value=ast.Name(id=en), attr='args'), slice=ast.Constant(value=0)), ops=[ast.Eq()], comparators=[ast.Constant(value=1)]), body=[i], orelse=[]))
        tb[1].handlers[0].body.extend(junk(en, len(olb) + 1))
        node.body = [ast.Assign(targets=[ast.Name(id=var)], value=ast.Constant(value=0), lineno=None)] + tb
        return node

    def on(node):
        if isinstance(node, ast.FunctionDef):
            return _bl(node)
        return node
    nb = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nb.append(on(node))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            nb.extend(bl([node]))
        elif isinstance(node, ast.Expr):
            nb.extend(bl([node]))
        else:
            nb.append(node)
    tree.body = nb
    return tree

def __moreobf(x):
    return ast.unparse(_moreobf(ast.parse(x)))

antipycdc = ''
for i in range(1000):
    antipycdc += f"你器(你器(你器(你器(你器(你器('')))))),"
antipycdc = "try:Anhnguyencoder=[" + antipycdc + "]\nexcept:pass"
ANTI_PYCDC = f"""
def 你器(你):
    return 你
try:pass
except:pass
finally:pass
{antipycdc}
finally:int(2011-2111)
"""
def mahoa(code: str):
    m  = __import__('marshal')
    lz = __import__('lzma')
    zl = __import__('zlib')
    bz = __import__('bz2')
    b64= __import__('base64')

    sexy = runtime_lock() + cink + anti + anti1 + anti2 + antirq + antiglb + antivm
    sexy1 = obf_var + d_var + ANTI_MEMORY + lolmemaythomlam
    sexy2 = sexy + sexy1

    minhanh = moreobf1(sexy2)

    tree = ast.parse(code)
    tree = Obf().visit(tree)
    ast.fix_missing_locations(tree)
    cuto = ast.unparse(tree)

    compiled = compile(cuto, "<FoNixA>", "exec")
    raw = m.dumps(compiled)
    fake = lol(raw)

    pairs = xor(fake)
    mapped_hex = phoenixa(fake.replace('\n', ''))
    enc2 = marshal_load(pairs, mapped_hex)

    code = minhanh + enc2
    code = ANTI_PYCDC + code

    no1 = ast.parse(speed(code))
    code = chimto(lop_gia(no1))
    final = __moreobf(code)
    code = final

    code = m.dumps(compile(code, '<FoNixA>', 'exec'))

    code = lz.compress(code)
    code = zl.compress(code)
    code = bz.compress(code)
    code = b64.b85encode(code)

    return code[::-1]

if __name__ == "__main__":
    from pystyle import Add,Center,Anime,Colors,Colorate,Write,System
    from sys import platform


    sys.setrecursionlimit(99999999)        
    ver = str(sys.version_info.major)+'.'+str(sys.version_info.minor)

    try:
        import random, marshal, base64, bz2, zlib, lzma, time, sys, builtins, ast, requests
        from pystyle import Add,Center,Anime,Colors,Colorate,Write,System
        from sys import platform
        from ast import *
        from pystyle import *
    except ModuleNotFoundError:
        print('>> Installing Module')
        __import__('os').system(f'pip{ver} install pystyle')
        from pystyle import *

    System.Clear()

    def clear():
        if platform[0:3]=='lin':
            os.system('clear')
        else:
            os.system('cls')
    meohieu = """
            █ ▄▄   ▄  █ ████▄ ▄███▄      ▄   ▄█     ▄  ██   
            █   █ █   █ █   █ █▀   ▀      █  ██ ▀▄   █ █ █  
            █▀▀▀  ██▀▀█ █   █ ██▄▄    ██   █ ██   █ ▀  █▄▄█ 
            █     █   █ ▀████ █▄   ▄▀ █ █  █ ▐█  ▄ █   █  █ 
             █       █        ▀███▀   █  █ █  ▐ █   ▀▄    █ 
              ▀     ▀                 █   ██     ▀       █  
                                                        ▀
                      PHONIX A OBF BETA FINAL
                  Code By AnhNguyenCoder (CteVcl)
                INFO AUTHOR >  FACEBOOK : www.facebook.com/anhnguyencoder.izumkonata
                INFO AUTHOR >  TELEGRAM : t.me/ctevclwar
                USER : ONE FILE (python PhoNixA.py), FOLDER (Enter File: <folder>)"""

    def banner():
        print('\x1b[0m',end='')
        clear()
        a=Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), meohieu)
        for i in range(len(a)):
            sys.stdout.write(a[i])
            sys.stdout.flush()

    banner()
    print()

    dark = Col.dark_gray
    light = Col.light_gray

    def stage(text: str, symbol: str = '>>', col1=None) -> str:
        if col1 is None:
            col1 = light
        return f" {Col.Symbol(symbol, col1, dark)} {text}{light}"

    def stage2(text: str, symbol: str = '...', col1=None) -> str:
        if col1 is None:
            col1 = light
        return f" {Col.Symbol(symbol, col1, dark)} {text}{light}"

    while True:
        file = input(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), "Enter File: "))).strip()
        if not file:
            print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.gray)), "File name cannot be empty!")))
            continue
        if not os.path.isfile(file):
            print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.gray)), "File not found! Please enter again.")))
            continue
        if not file.lower().endswith(".py"):
            print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.gray)), "Only .py files are allowed!")))
            continue
        break

    print(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), '-----------------------------------------------------------------'))
    print(stage2(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), 'Start Encode...')))
    st = time.time()

    import ast
    from ast import *

    def rb2():
        return ''.join(random.choices([chr(i) for i in range(12356, 12544) if chr(i).isprintable() and chr(i).isidentifier()], k=11))
    
    cust = rb2()

    with open(file, 'r', encoding='utf-8') as f:
        code = ast.parse(f.read())

    def vip(s, junk=f"{cust}/3�{cust}.�.4{cust}c,2�__AnhNguyenCoder___1.5�{cust}.{cust}.6767�$@.42__AnhNguyenCoder___{cust}�{cust}..011�.20{cust}.12{cust}4.�1.6{cust}", max_=3):
        import random
        random.seed()
        __map__ = {}
        _fmt_ = []

        for i, c in enumerate(s):
            key = junk + str(i)
            __map__[key] = c
            _fmt_.append(f"%({key})s")

            for _ in range(random.randint(1, max_)):
                __map__[junk + hex(random.getrandbits(1))] = junk

        fmt = ''.join(_fmt_)
        return f"('{fmt}' % {__map__})"

    print(stage2(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), 'Converting F-String To Join String...')))
    
    def rb():
        return ''.join(random.choices([chr(i) for i in range(44032, 55204) if chr(i).isprintable() and chr(i).isidentifier()], k=11))

    string = rb()

    payload = mahoa(code)
    print(stage2(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), 'Hide Builtins...')))
    load = 76,111,97,100,105,110,103,46,46,46,13

    lobby = f"""#!/bin/python{ver}
# -*- coding: utf-8 -*-
__OBF__ = ('FonixA')
__Author__ = ('Anhnguyencoder')
__Tele__ = ('https://t.me/ctevclwar')
__In4__ = ('https://www.facebook.com/anhnguyencoder.izumkonata')
__CMT__ = {{
    "EN": "Việc sử dụng obf này để lạm dụng mục đích xấu, người sở hữu sẽ không chịu trách nghiệm!",
    "VN": "Using this obf for bad purposes, the owner will not be responsible!"
}}

if str(__import__("sys").version_info.major)+"."+str(__import__("sys").version_info.minor) != "{ver}":
    print(f'>> Your Python Version Is {{str(__import__("sys").version_info.major)+"."+str(__import__("sys").version_info.minor)}}.\\n>> Please Install Python {ver} To Run This File!')
    __import__('sys').exit()
else:
    getattr(__import__('sys').stdout,'write')(''.join(map(chr, {load})))

try:exec(__import__{vip('marshal')}.loads(__import__{vip('lzma')}.decompress(__import__{vip('zlib')}.decompress(__import__{vip('bz2')}.decompress(__import__{vip('base64')}.b85decode({payload}[::-1]))))),globals())
except Exception as {string}:
    print({string})
except KeyboardInterrupt:pass"""

    print(stage2(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), 'Compiling...')))

    def color_loading():
        duration = 2.0
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            percent = 100.0 if elapsed >= duration else 1.0 + elapsed / duration * 99.0
            text = f'>> Encoding... {percent:09.6f}%'
            colored = Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), text)
            sys.stdout.write('\r' + colored)
            sys.stdout.flush()
            if percent >= 100.0:
                break
            time.sleep(0.01)
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    color_loading()

    out = "obf-" + file
    open(out, "w", encoding="utf-8").write(lobby)
    print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)), f'Execution Time {time.time()-st:.3f}s')))
    print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)),f"Encode Success -> {out}")))
    size_kb = os.path.getsize(out) / 1024
    print(stage(Colorate.Diagonal(Colors.DynamicMIX((Col.red, Col.orange)),f'Output file size {size_kb:.2f} KB')))
