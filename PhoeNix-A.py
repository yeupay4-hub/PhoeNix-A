try:
    import ast, random, marshal, base64, lzma, time, sys, os, sys, string
    import string as _string
    from pystyle import Add,Center,Anime,Colors,Colorate,Write,System
    from sys import platform
    from ast import *
except Exception as e:
    print(e)

mupvailol = r"""
list_func = ['print', 'open']
hooked_funcs = {}
inspect = __import__('inspect')
def hooknoi(func_name, *args, **kwargs):
    checked_paths = set()
    for frame in inspect.stack():
        filename = frame.filename
        if filename not in checked_paths:
            checked_paths.add(filename)
            try:
                module_names = ['requests', 'pystyle', 'ssl', 'socket', 'inspect', 'urllib']
                for module_name in module_names:
                    try:
                        if "requests" in module_name:
                            requests_path = __import__('os').path.dirname(__import__('os').path.abspath(__import__('requests').__file__))
                            api_path = __import__('os').path.join(requests_path, 'api.py')
                            if api_path in frame.filename:
                                try:__import__('sys').exit(f"Lỗi , Hãy Cài Lại Thư Viện Requests")
                                except:pass
                                finally:__import__('sys').exit(f"Lỗi , Hãy Cài Lại Thư Viện Requests")
                        module_path = __import__('os').path.abspath(__import__(module_name).__file__)
                        if module_path in filename:
                            try:__import__('sys').exit(f"Lỗi , Hãy Cài Lại Thư Viện {api_path}")
                            except:pass
                            finally:__import__('sys').exit(f"Lỗi , Hãy Cài Lại Thư Viện {api_path}")
                    except (ImportError, AttributeError,ModuleNotFoundError):
                        continue
            except Exception:
                pass
    hooked_funcs[func_name](*args, **kwargs)
def chimcubu():
    for func_name in list_func:
        try:
            func = getattr(__builtins__, func_name)
            hooked_funcs[func_name] = func
            setattr(__builtins__, func_name, lambda *args, func_name=func_name, **kwargs: hooknoi(func_name, *args, **kwargs))
        except AttributeError:
            raise MemoryError
def nohook():
    for func_name, original_func in hooked_funcs.items():
        setattr(__builtins__, func_name, original_func)
    hooked_funcs.clear()
chimcubu()
import inspect
nohook()
"""

dz=  mupvailol.encode()

hookcc = f'''

exec({dz})

# '''

def antibypass():

    def anti(s: str, kkk=69):

        def f(n):
            a, b = (n & 240, n & 15)
            return f'(({a + 10000000000000000000000000}) >>  ({b + 100000000000000000000000000000000000}))' if n > 15 else str(n)
        fx = [f(ord(c) ^ kkk) for c in s]
        mm = ', '.join(fx)
        return f"""((lambda __Anhnguyencoder__: __Anhnguyencoder__(*[__dat__('Biet Dzai Roi',{mm})]))(lambda *__occak__: ((lambda __thknqu__, __Anhnguyencoder__:__Anhnguyencoder__().join([*map(lambda n: __Anhnguyencoder__().format((n ^ 64)), __Anhnguyencoder__)]))(lambda: getattr(''.__class__, '__add__')('__Anhnguyencoder__', ''), lambda: "__CONCAC__"))))"""
    junk_code = []
    for i in range(80):
        junk_code.append(f'\ndef _junk{i}():\n    x = {random.randint(1000, 9999)}\n    for n in range(50):\n        x ^= (n << {i % 5})\n    return x\n')
    fake_flow = []
    for i in range(80):
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
        for _ in range(20):
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
        for i in range(80):
            antipycdc += f"__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__(__Anhnguyencoder__('{d}')))))),"
        antipycdc = "try:anhnguyen=[" + antipycdc + c + "]\nexcept:pass"
        text = f"""
{''.join(junk_code)}
def __CTEVCLDZAI__(__chanankdi__):
    return __chanankdi__
try:pass
except:pass
finally:pass
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
    lam3 = ast.Lambda(args=_args(rb()), body=ast.Call(func=ast.Attribute(value=ast.Constant(value=''), attr='join', ctx=ast.Load()), args=[ast.GeneratorExp(elt=ast.Call(ast.Name('chr', ast.Load()), [ast.Name(v, ast.Load())], []), generators=[ast.comprehension(target=ast.Name(v, ast.Store()), iter=ast.List([ast.Constant(x) for x in lst], ast.Load()), ifs=[], is_async=0)])], keywords=[]))
    lam2 = ast.Lambda(_args(rb()), ast.Call(lam3, [ast.Constant('AnhNguyenCoder')], []))
    lam1 = ast.Lambda(_args(rb()), ast.Call(lam2, [ast.Constant('AnhNguyenCoder')], []))
    return ast.Call(lam1, [ast.Constant('AnhNguyenCoder')], [])

def obfint(i):
    haha = 2010 - i
    lam3 = ast.Lambda(_args(rb()), ast.BinOp(ast.Constant(2010), ast.Sub(), ast.Constant(haha)))
    lam2 = ast.Lambda(_args(rb()), ast.Call(lam3, [ast.Constant('AnhNguyenCoder')], []))
    lam1 = ast.Lambda(_args(rb()), ast.Call(lam2, [ast.Constant('AnhNguyenCoder')], []))
    return ast.Call(lam1, [ast.Constant('AnhNguyenCoder')], [])

def obffloat(f):
    return ast.Call(func=ast.Name('float', ast.Load()), args=[obfstr(repr(f))], keywords=[])

def obfstring(v):
    if isinstance(v, bool):
        return ast.Constant(value=v)
    if isinstance(v, str):
        return obfstr(v)
    if isinstance(v, int):
        return obfint(v)
    if isinstance(v, float):
        return obffloat(v)
    return ast.Constant(value=v)

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
                value_expr = Call(func=Name(id='str', ctx=Load()), args=[value_expr], keywords=[])
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
                        spec_value = Call(func=Name(id='str', ctx=Load()), args=[spec_value], keywords=[])
                    elif i.format_spec.conversion == 114:
                        spec_value = Call(func=Name(id='repr', ctx=Load()), args=[spec_value], keywords=[])
                    elif i.format_spec.conversion == 97:
                        spec_value = Call(func=Name(id='ascii', ctx=Load()), args=[spec_value], keywords=[])
                    spec_expr = spec_value
                else:
                    spec_expr = i.format_spec
                value_expr = Call(func=Name(id='format', ctx=Load()), args=[value_expr, spec_expr], keywords=[])
            elif i.conversion == -1:
                value_expr = Call(func=Name(id='str', ctx=Load()), args=[value_expr], keywords=[])
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
        if isinstance(node.value, (str, int, float)):
            node = obfstring(node.value)
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
        core = ast.Call(func=ast.Name('getattr', ast.Load()), args=[ast.Call(func=ast.Name('__import__', ast.Load()), args=[ast.Constant('builtins')], keywords=[]), ast.Constant(name)], keywords=[])
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

def ast_lol(code):
    if isinstance(code, list):
        code = ast.Module(body=code, type_ignores=[])
    elif isinstance(code, str):
        code = ast.parse(code)

    meo = AntiSafeVarSpam().visit(code)
    code = AntiSafeNoise().visit(meo)
    code = Vars().visit(code)

    ast.fix_missing_locations(code)
    return ast.unparse(code)

from ast import *

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

        def __init__(self):
            self.in_fstring = False

        def visit_JoinedStr(self, node):
            old = self.in_fstring
            self.in_fstring = True
            self.generic_visit(node)
            self.in_fstring = old
            return node

        def visit_Constant(self, node):
            if self.in_fstring:
                return node
            if isinstance(node.value, str) and node.value and ('|' not in node.value) and (not any((e in node.value for e in lolrong))):
                hx = node.value.encode().hex()
                if not all((c in string for c in hx)):
                    return node
                mapped = ''.join((enc[c] for c in hx))
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
    if isinstance(code, str):
        tree = ast.parse(code)
    elif isinstance(code, list):
        tree = ast.Module(body=code, type_ignores=[])
    else:
        tree = code
    enc = dict(zip(string, lolrong))

    class StringBreaker(ast.NodeTransformer):

        def __init__(self):
            self.in_fstring = False

        def visit_JoinedStr(self, node):
            old = self.in_fstring
            self.in_fstring = True
            self.generic_visit(node)
            self.in_fstring = old
            return node

        def visit_Constant(self, node):
            if self.in_fstring:
                return node
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
                    e += '\u200b'
                out.append(e)
            mapped = ''.join(out)
            return ast.Call(func=ast.Name(id='anhnguyencoder', ctx=ast.Load()), args=[ast.Constant(mapped)], keywords=[])
    tree = StringBreaker().visit(tree)
    ast.fix_missing_locations(tree)
    return tree

string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
cust = '🧸🐲⭐✦✧✨💫🎀👄👁️🔱💢☄️🌪❄️🌀🥋🥊⚔️👊🙌👐🟠🔴🟡🟢🔵🟣⚫⚪👽🤖👺🐢🐒🦍👑💎🔮🍑🍗🍚🍶🏯⛩⛰🛡👑🧙\u200d♂️🤜🤛😡😤🥵🤯🌌🌍🌑☀️🌠'

memaybeo = dict(zip(string, cust))
vuto = {v: k for k, v in memaybeo.items()}
memaybeo_bi = {ch: em for ch, em in memaybeo.items() if vuto.get(em) == ch}

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

def obfstring_runtime() -> str:
    return (
        "def anhnguyencoder(s):\n"
        f"    dec = {vuto!r}\n"
        "    hx = []\n"
        "    i = 0\n"
        "    while i < len(s):\n"
        "        for e, h in dec.items():\n"
        "            if s.startswith(e, i):\n"
        "                hx.append(h)\n"
        "                i += len(e)\n"
        "                break\n"
        "        else:\n"
        "            i += 1\n"
        "    return bytes.fromhex(''.join(hx)).decode('utf-8')\n"
    )

_str = "str"
def _rd():
    return "".join(__import__("random").sample([chr(i) for i in range(97, 122)], k=5))

def randomint():
    return "".join(__import__("random").sample([str(i) for i in range(1, 20)], k=5))

def random_match_case():
    var1 = ast.Constant(value=randomint(), kind=None)
    var2 = ast.Constant(value=randomint(), kind=None)
    return ast.Match(
        subject=ast.Compare(
            left=var1,
            ops=[ast.Eq()],
            comparators=[var2],
        ),
        cases=[
            ast.match_case(
                pattern=ast.MatchSingleton(value=True),
                body=[
                    ast.Assign(
                        lineno=0,
                        col_offset=0,
                        targets=[],
                        value=[ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="MemoryError", ctx=ast.Load()),
                        args=[ast.Constant(value=True)], keywords=[],
                    ),)],
                    )
                ],
            ),
            ast.match_case(
                pattern=ast.MatchSingleton(value=False),
                body=[
                    ast.Assign(
                        lineno=0,
                        col_offset=0,
                        targets=[ast.Name(id=_rd(), ctx=ast.Store())],
                        value=ast.Constant(value=[[True], [False]], kind=None),
                    ),
                    ast.Expr(
                        lineno=0,
                        col_offset=0,
                        value=ast.Call(
                            func=ast.Name(id=_str, ctx=ast.Load()),
                            args=[ast.Constant(value=[_rd()], kind=None)],
                            keywords=[],
                        ),
                    ),
                ],
            ),
        ],
    )


def trycatch(body, loop):
    ar = []
    for x in body:
        j = x
        for _ in range(5):
            j = ast.Try(
                body=[random_match_case(),

                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="MemoryError", ctx=ast.Load()),
                        name=_rd(),
                        body=[j],
                    )
                ],
                orelse=[],
                finalbody=[],
            )
            j.body.append(
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="MemoryError", ctx=ast.Load()),
                        args=[ast.Constant(value=True)], keywords=[],
                    ),
                    cause=None,
                )
            )
        ar.append(j)
    return ar

def random_match_case1():
    var1 = ast.Constant(value=randomint(), kind=None)
    var2 = ast.Constant(value=randomint(), kind=None)
    return ast.Match(
        subject=ast.Compare(
            left=var1,
            ops=[ast.Eq()],
            comparators=[var2],
        ),
        cases=[
            ast.match_case(
                pattern=ast.MatchSingleton(value=True),
                body=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="MemoryError", ctx=ast.Load()),
                            args=[ast.Constant(value=True)], keywords=[],
                        ),
                    ),
                ],
            ),
            ast.match_case(
                pattern=ast.MatchSingleton(value=False),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=_rd(), ctx=ast.Store())],
                        value=ast.List(elts=[ast.List(elts=[ast.Constant(value=True, kind=None)], ctx=ast.Load()), ast.List(elts=[ast.Constant(value=False, kind=None)], ctx=ast.Load())], ctx=ast.Load()),
                    ),
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id=_str, ctx=ast.Load()),
                            args=[ast.List(elts=[ast.Constant(value=_rd(), kind=None)], ctx=ast.Load())],
                            keywords=[],
                        ),
                    ),
                ],
            ),
        ],
    )


def trycatch1(body, loop):
    ar = []
    for x in body:
        j = x
        for _ in range(5):
            j = ast.Try(
                body=[random_match_case1(),

                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="MemoryError", ctx=ast.Load()),
                        name=_rd(),
                        body=[j],
                    )
                ],
                orelse=[],
                finalbody=[],
            )
            j.body.append(
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="MemoryError", ctx=ast.Load()),
                        args=[ast.Constant(value=True)], keywords=[],
                    ),
                    cause=None,
                )
            )
        ar.append(j)
    return ar


import ast, random

def moreobf1(src: str) -> str:
    tree = ast.parse(src)

    def rd():
        return '__0x_' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        
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
for i in range(60):
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
import ast, random

class StringObfuscator(ast.NodeTransformer):

    def __init__(self):
        self.in_fstring = False

    def build_xor(self, s):
        key = random.randint(1, 255)
        data = [ord(c) ^ key for c in s]
        return ast.parse(f"(lambda k: ''.join(chr(x ^ k) for x in {data}))({key})").body[0].value

    def build_add(self, s):
        key = random.randint(1, 50)
        data = [ord(c) + key for c in s]
        return ast.parse(f"(lambda k: ''.join(chr(x - k) for x in {data}))({key})").body[0].value

    def build_reverse_xor(self, s):
        key = random.randint(1, 255)
        data = [ord(c) ^ key for c in s[::-1]]
        return ast.parse(f"(lambda k: ''.join(chr(x ^ k) for x in {data})[::-1])({key})").body[0].value

    def build_chr_join(self, s):
        data = [ord(c) for c in s]
        return ast.parse(f"''.join(map(chr,{data}))").body[0].value

    def build_slice_mix(self, s):
        data = s.encode().hex()
        return ast.parse(f"bytes.fromhex('{data}').decode()").body[0].value

    def visit_JoinedStr(self, node):
        old = self.in_fstring
        self.in_fstring = True
        self.generic_visit(node)
        self.in_fstring = old
        return node

    def visit_Constant(self, node):
        if self.in_fstring:
            return node
        if not isinstance(node.value, str):
            return node
        s = node.value
        if not s:
            return node
        if any(e in s for e in lolrong):
            return node
        dangerous = {'__file__', '__name__', '__main__', 'exec', 'eval', 'compile', 'marshal', 'builtins', 'sys', 'os'}
        if s in dangerous:
            return node
        parent = getattr(node, 'parent', None)
        if isinstance(parent, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if parent.body and parent.body[0] is node:
                return node
        builders = [self.build_xor, self.build_add, self.build_reverse_xor, self.build_chr_join, self.build_slice_mix]
        try:
            new_node = random.choice(builders)(s)
            return ast.copy_location(new_node, node)
        except Exception:
            return node

from ast import *
def _repair_locations(node):
    for n in ast.walk(node):
        for attr, minv in (
            ('lineno',         1),
            ('end_lineno',     1),
            ('col_offset',     0),
            ('end_col_offset', 0),
        ):
            if attr in n._attributes:
                v = getattr(n, attr, None)
                if not isinstance(v, int) or v < minv:
                    try:
                        delattr(n, attr)
                    except AttributeError:
                        pass
    ast.fix_missing_locations(node)
    return node


def lop_gia(code):
    if isinstance(code, list):
        code = ast.Module(body=code, type_ignores=[])
    if isinstance(code, ast.AST):
        _repair_locations(code) 
        co = compile(code, "<FoNixA>", "exec")
    else:
        co = compile(ast.parse(code), "<FoNixA>", "exec")

    d = marshal.dumps(co)
    k = random.randint(1, 255)
    d = bytes(b ^ k for b in d)
    d = base64.b85encode(lzma.compress(d))
    return """
try:
    TinhYeu100ThapKy=['PyMarshal_ReadObjectFromString', 'PyEval_EvalCode']
except:
    pass
else:
    pass
if any(bytes((__import__('ctypes').c_ubyte * 2).from_address(__import__('ctypes').cast(getattr(__import__('ctypes').pythonapi, DJTTHGNGU), __import__('ctypes').c_void_p).value)) == bytes([255, 37]) for DJTTHGNGU in TinhYeu100ThapKy):raise MemoryError('AnhNguyen...')
BaDaoCucGayNha = __import__('marshal').loads(bytes(djtmenhamay ^ %d for djtmenhamay in __import__('lzma').decompress(__import__('base64').b85decode(%r))))
try:__import__('types').FunctionType(BaDaoCucGayNha, globals())()
except:pass
""" % (k, d)

def mahoa(code: str):
    m   = __import__('marshal')
    zl  = __import__('zlib')
    b64 = __import__('base64')

    sexy = cink + anti + anti1 + anti2
    sexy1 = d_var + ANTI_MEMORY
    um = sexy + runtime_lock() + sexy1
    minhanh = moreobf1(um + hookcc)
    tree = ast.parse(code)
    tree = Obf().visit(tree)
    ast.fix_missing_locations(tree)
    cuto = ast.unparse(tree)
    code = ast.parse(minhanh + '\n' + cuto)
    code = ast.unparse(code)
    code = ANTI_PYCDC + code
    no1 = ast.parse(speed(code))
    _d  = ast.parse(obfstring_runtime())
    no1 = ast.Module(body=_d.body + no1.body, type_ignores=[])
    ast.fix_missing_locations(no1)
    no1 = chimto(no1)
    tree = StringObfuscator().visit(no1)
    no1 = trycatch1(tree.body, 1)
    mod = ast.Module(body=no1, type_ignores=[])
    _repair_locations(mod)
    no1_src = lop_gia(mod)
    code = ast.parse(no1_src)
    code = trycatch(code.body, 1)
    code = ast_lol(code)
    tree = StringObfuscator().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    final = __moreobf(code)
    final = obfstring_runtime() + final
    code = ast.parse(final)
    code = m.dumps(compile(ast.unparse(code), '<FoNixA>', 'exec'))
    code = zl.compress(code, level=5)
    code = b64.b85encode(code)

    return code[::-1]

if __name__ == "__main__":
    from pystyle import Add,Center,Anime,Colors,Colorate,Write,System
    from sys import platform
    
    if sys.version_info[:2] == (3, 9):
        raise RuntimeError("Python 3.9 is not supported. Please use Python 3.10+.")
        
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

    caklol = "__Anhnguyencoder__"
    bien = "__Anhnguyencoder__=vars(__import__('builtins'))['e'+'x'+'e'+'c']"

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
{bien}
try:{caklol}(__import__{vip('marshal')}.loads(__import__{vip('zlib')}.decompress(__import__{vip('base64')}.b85decode({payload}[::-1]))),globals())
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
