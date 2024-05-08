import argparse, subprocess, sys, os, re, math, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc, inspect, signal, traceback
from importlib.machinery import SourceFileLoader
from itertools import chain, groupby, islice, product, permutations, combinations
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict, Counter
try: # Optional debugging tool
    import q as qq
except:
    pass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy
import scipy.stats
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mapf

def build_mapf_config(a, use_lns=None):
    use_lns = use_lns or bool(a.get('num_subset_agents') or a.get('num_subset_agents_fn'))
    config = mapf.LNSConfig() if use_lns else mapf.Config()
    config.map_file = Path(__file__).parent / 'maps' / a.map
    config.num_agents = a.num_agents
    config.verbosity = a.verbosity
    config.use_cpp_rng = a.use_cpp_rng
    config.wait_at_goal = a.wait_at_goal
    if a.wait_at_goal: assert a.num_windows == 1
    config.max_solution_time = a.max_solution_time
    config.max_path_plan_time = a.max_path_plan_time
    a.seed and config.set_seed(a.seed)
    if use_lns:
        config.iterations = np.iinfo(np.int32).max if np.isinf(a.lns_iterations) else a.lns_iterations
        config.time_limit = a.get('time_limit', np.inf)
        config.occupancy = a.occupancy
        config.subset_fn = a.subset_fn if isinstance(a.get('subset_fn'), str) else ''
        config.num_subsets = a.num_subsets
        config.num_subset_agents = a.num_subset_agents if isinstance(a.num_subset_agents, int) else 0
        config.subsolver = a.solver
        config.acceptance_threshold = a.acceptance_threshold
    return config

def build_mapf_config_graph(a, use_lns=None):
    config = build_mapf_config(a, use_lns=use_lns)
    return config, (mapf.SortationGraph(config) if a.map == 'sorting_map.grid' else mapf.UndirectedGraph(config))

def setup_mapf(a):
    a.config, a.G = build_mapf_config_graph(a)
    a.obstacles = a.G.get_obstacles()
    a.obstacle_rows = a.obstacles // a.G.cols
    a.obstacle_cols = a.obstacles % a.G.cols
    if a.get('prepool', 1) > 1:
        a.G_orig = a.G
        a.G = Namespace(rows=(a.G.rows - 1) // a.prepool + 1, cols=(a.G.cols - 1) // a.prepool + 1)
        a.G.size = a.G.rows * a.G.cols
        a.obstacle_rows = a.obstacle_rows // a.prepool
        a.obstacle_cols = a.obstacle_cols // a.prepool
        a.obstacles = a.obstacle_rows * a.G.cols + a.obstacle_cols

stats_header = ['window', 'num_finished_tasks', 'cost', 'initial_cost', 'shortest_cost', 'runtime', 'initial_runtime', 'LL_num_expanded', 'LL_num_generated']
pbs_stats_header = ['num_LL', 'HL_num_expanded', 'HL_num_generated']
full_pbs_stats_header = [*pbs_stats_header, 'runtime_occupancy', 'runtime_plan_paths', 'runtime_copy_priorities', 'runtime_find_conflicts', 'runtime_find_consistent_paths', 'runtime_find_replan_agents']

def get_stats_header(sol):
    return stats_header + (full_pbs_stats_header if isinstance(sol, mapf.PBSSolution) else [])
def get_full_pbs_stats(sol):
    return [getattr(sol, x) for x in full_pbs_stats_header] if isinstance(sol, mapf.PBSSolution) else []

substats_header = ['window', 'iteration', 'initial_cost', 'final_cost', 'runtime', 'LL_num_expanded', 'LL_num_generated', 'subset_fn', 'num_subset_agents', 'subset_fn_time']
def get_substats_header(subsol):
    x = substats_header.index('LL_num_generated') + 1
    return substats_header[:x] + (pbs_stats_header if isinstance(subsol, mapf.PBSSolution) else []) + substats_header[x:]
def get_pbs_stats(sol):
    return [getattr(sol, x) for x in pbs_stats_header] if isinstance(sol, mapf.PBSSolution) else []

def get_runtime_stats_header(is_pbs):
    keys = ['LL_num_expanded', 'LL_num_generated']
    return ['HL_num_expanded', 'HL_num_generated', 'num_LL', *keys] if is_pbs else keys

def get_runtime_stats(sol):
    return [getattr(sol, x) for x in get_runtime_stats_header(isinstance(sol, mapf.PBSSolution))]

def load_runtime_map(dir, name):
    start_dir = dir
    while dir.parent:
        if (runtime_coefs := dir / 'runtimes' / name).exists():
            return {(r['solver'], r['subset_fn'], r['num_subset_agents']): (r['w'], r['b']) for r in runtime_coefs.load()}
        dir = dir.parent
    raise RuntimeError(f'Could not find runtime file runtimes/{name} in parents of {start_dir}')

def bootstrap(values, num_resample=1000, confidence=95, geometric=False):
    if geometric: values = np.log(values)
    mean = np.nanmean(values, axis=0)
    indices = np.random.choice(len(values), size=num_resample * len(values), replace=True).reshape(num_resample, len(values))
    values = np.nanmean(values[indices, ...], axis=1)
    if geometric: mean, values = np.exp(mean), np.exp(values)
    lower = np.percentile(values, (100 - confidence) / 2, axis=0)
    upper = np.percentile(values, 100 - (100 - confidence) / 2, axis=0)
    return mean, lower, upper

def pack(sequences, axis=0): # Concatenate arrays with -1 as delimiter
    delimiter = [-1]
    return np.concatenate([x for y in sequences for x in [y, delimiter]], axis=axis)

def unpack(sequence, delimiter=-1): # Split "-1"-padded concatenated sequence into separate arrays
    return np.array([r[:-1] if r[-1] == delimiter else r for r in np.split(sequence, np.where(sequence == delimiter)[0] + 1) if len(r)], dtype=object)

def sif(keep, str):
    return str if keep else ''

def lif(keep, *x):
    return x if keep else []

def dif(keep, **kwargs):
    return kwargs if keep else {}

def flatten(x):
    return [z for y in x for z in y]

def split(x, sizes):
    return np.split(x, np.cumsum(sizes[:-1]))

def recurse(x, fn):
    if isinstance(x, dict):
        return type(x)((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(recurse(v, fn) for v in x)
    return fn(x)

def pad_arrays(arrs, value, length=None):
    length = length or max(len(x) for x in arrs)
    return np.array([np.concatenate([x, np.full(length - len(x), value)]) for x in arrs])

def trim_arrays(arrs, length=None):
    length = length or min(len(x) for x in arrs)
    return np.array([x[:length] for x in arrs])

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def gsmooth(y, sigma):
    if sigma == 0: return y
    from scipy.ndimage.filters import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma)

def shell(cmd, wait=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    stdout = stdout or subprocess.DEVNULL
    stderr = stderr or subprocess.DEVNULL
    if not isinstance(cmd, str):
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr, env=os.environ)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode().rstrip('\n') if out else '', err.decode().rstrip('\n') if err else ''

def git_state(dir=None):
    cwd = os.getcwd()
    dir = dir or shell('git rev-parse --show-toplevel')[0]
    os.chdir(dir)
    status = shell('git status')[0]
    base_commit = shell('git rev-parse HEAD')[0]
    diff = shell('git diff %s' % base_commit)[0]
    os.chdir(cwd)
    return dict(base_commit=base_commit, diff=diff, status=status)

class Stop(Exception):
    def _render_traceback_(self):
        pass

def str2num(s):
    try: return int(s)
    except:
        try: return float(s)
        except: return s

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def to_hashable(x):
    if isinstance(x, dict):
        return HashableDict((k, to_hashable(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(v) for v in x)
    return x

def parse_options(defs, *options, skip_defaults={}):
    """
    Each option takes the form of a string keyvalue or tuple of string keyvalue and definition. Match keyvalue by the following precedence in defs
    defs: {
        keyvalue: {config_key: config_value, ...},
        key: None, # implicitly {key: value}
        key: config_key, # implicitly {config_key: value}
        key: v -> {config_key: config_value, ...},
        ...
    }
    options: [key1value1, key2value2_key3value3, ...]
    """
    options = flatten([x.split('_') if isinstance(x, str) else [x] for x in options if x])
    name_fragments = []
    kwargs = {}
    def update_recursive(kwargs, change):
        for k, v in change.items():
            if isinstance(v, dict):
                v_ = kwargs.setdefault(k, {})
                assert isinstance(v_, dict)
                update_recursive(v_, v)
            else:
                assert k not in kwargs
                kwargs[k] = v

    for o in options:
        change = {}
        if not isinstance(o, str):
            change.update(o[1])
        elif o in defs:
            change.update(defs[o])
        else:
            k, v = re.match('([a-zA-Z]*)(.*)', o).groups()
            fn_str_none = defs[k]
            v = str2num(v)
            if fn_str_none is None:
                change.update({k: v})
            elif isinstance(fn_str_none, str):
                change.update({fn_str_none: v})
            else:
                change.update(fn_str_none(v))
        n = o if isinstance(o, str) else o[0]
        for k, v in change.items():
            if k not in skip_defaults or skip_defaults[k] != v:
                name_fragments.append(n)
                update_recursive(kwargs, change)
                break
    name = '_'.join(name_fragments)
    return name, kwargs

def sbatch(cpu=1, gpu=False):
    return f"""#!/bin/sh

#SBATCH -o output-%j.log
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c {cpu}                     # number of cpu per task
{f'#SBATCH --gres=gpu:volta:{int(gpu)}' if gpu else '#SBATCH --constraint xeon-p8'}
"""

def save_submit(name, *args, cmd, cpu=1, gpu=False, cd=None, print_only=False, dependency=None):
    sub_path = '_'.join(map(str, [name, *args])) + '.sh'
    print_cmd = cmd if print_only else f'sbatch{sif(dependency, f" --dependency afterok:{dependency}")} {sub_path}'
    if cd:
        print_cmd = f'cd {cd} && {print_cmd}'
        sub_path = cd / sub_path
    if print_only:
        print(print_cmd)
        return
    content = '\n'.join([sbatch(cpu=cpu, gpu=gpu), cmd, ''])
    with open(sub_path, 'w') as f:
        f.write(content)
    print(print_cmd) # Note the printed command!

def load_config(dir, parent=0):
    path = dir / 'config.yaml'
    conf = Namespace(path.load() if path.exists() else {})
    if parent <= 0: return conf
    return load_config(dir.real.parent, parent - 1).var(**conf)

def build_flags(vars, *args, **kwargs):
    def build_flag(k, v):
        if isinstance(v, bool):
            return sif(v, f'--{k}')
        return f'--{k} {v}'
    flags = [build_flag(arg, vars[arg]) for arg in args]
    flags.extend(build_flag(key, value) for key, value in kwargs.items())
    return ' '.join(flag for flag in flags if flag)

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def convert_yaml(x):
    if isinstance(x, Path):
        return x.str
    elif isinstance(x, dict):
        return dict((k, convert_yaml(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return list(map(convert_yaml, x))
    elif isinstance(x, np.generic):
        return x.item()
    return x

def format_yaml(dict_):
    return yaml.dump(convert_yaml(dict_))

def load_text(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def extract(input_path, output_path=None):
    if input_path[-3:] == '.gz':
        if not output_path:
            output_path = input_path[:-3]
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise RuntimeError('Don\'t know file extension for ' + input_path)

def restore(a, net, opt=None, scaler=None, net_key='net'):
    if a.checkpoint_step is None:
        saved_models_paths = (a.train_dir / 'models').glob('*.pth')
        if len(saved_models_paths):
            ckpt_steps = [int(x.name.stem) for x in saved_models_paths]
            a.checkpoint_step = max(ckpt_steps)
    if a.checkpoint_step is not None:
        a.checkpoint_path = a.train_dir / 'models' / a.checkpoint_step + '.pth'
    if a.get('checkpoint_path') is not None:
        checkpoint = torch.load(a.checkpoint_path, map_location=a.device)
        net.load_state_dict(checkpoint[net_key])
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])
            for i in range(len(opt.param_groups)):
                opt.param_groups[i]['lr'] = a.lr
                opt.param_groups[i]['initial_lr'] = a.lr
        a.use_amp and scaler is not None and scaler.load_state_dict(checkpoint['scaler'])
        print(f'Loaded checkpoint at step={checkpoint["step"]} from {a.checkpoint_path}')
        return checkpoint['step'], time() - checkpoint['time']
    print(f'Initialized network at step=0')
    return 0, time()

class Logger:
    def __init__(self, base_name, start_time=None):
        self.start_time = start_time or time()
        self.save_path = base_name + f'_{datetime.now().isoformat(timespec="seconds")}.ftr'
        self.results = []

    def log(self, step, result):
        total_time = time() - self.start_time
        result = {k: v for k, v in result.items() if k.startswith('_') or v is not None}
        result['total_time'] = total_time

        is_delim = lambda k: k[0].startswith('_')
        result_groups = [list(group) for is_del, group in groupby(result.items(), is_delim) if not is_del]
        line_width = 160
        pre = ' | '.join([datetime.now().isoformat(timespec='seconds'), f'step {step}'])
        pre_w = len(pre)
        shortest_num = lambda v: str1 if len(str1 := f'{v}') <= len(str2 := f'{v:.3g}') else str2
        for group in result_groups:
            stats = [f'{k} {v}' if isinstance(v, str) else f'{k} {shortest_num(v)}' for k, v in group]
            curr_w = pre_w + 3
            i_start = 0
            for i, w in enumerate(map(len, stats)):
                if curr_w + w > line_width:
                    print(' | '.join([pre, *stats[i_start:]]))
                    i_start = i
                    curr_w, pre = pre_w + 3, ' ' * pre_w
            print(' | '.join([pre, *stats[i_start:]]))
            pre = ' ' * pre_w
        sys.stdout.flush()
        result = dict(step=step, **{k: v for k, v in result.items() if not k.startswith('_')})
        self.results.append(result)
    
    def sublog(self, step, substep, result):
        self.results.append(dict(step=step, substep=substep, **result))

    def save(self):
        df = pd.DataFrame(self.results)
        df = df.astype({k: np.float16 for k in df.columns if k not in ['step', 'substep']})
        df.to_feather(self.save_path)

def to_torch(x, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
    def helper(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device=device, **kwargs)
        elif np.isscalar(x):
            return x
        return torch.from_numpy(x).to(device=device, **kwargs)
    return recurse(x, helper)

def from_torch(t, force_scalar=False):
    def helper(t):
        if not isinstance(t, torch.Tensor):
            return t
        x = t.detach().cpu().numpy()
        if force_scalar and (x.size == 1 or np.isscalar(x)):
            return np.asscalar(x)
        return x
    return recurse(t, helper)

def compute_output_shape(input_shape, layers):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            *dims, in_features = input_shape
            assert in_features == layer.in_features
            input_shape = (*dims, layer.out_features)
            continue
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            in_channels, *dims = input_shape
            assert in_channels == layer.in_channels
            out_channels = layer.out_channels
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            in_channels, *dims = input_shape
            out_channels = in_channels
        else:
            continue
        dims = 1 + (np.array(dims) + 2 * np.array(layer.padding) - np.array(layer.dilation) * (np.array(layer.kernel_size) - 1) - 1) // np.array(layer.stride)
        input_shape = (out_channels, *dims)
    return input_shape

def recurse(x, fn):
    if isinstance(x, dict):
        return type(x)((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(recurse(v, fn) for v in x)
    return fn(x)

class Namespace(dict):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        kvs = dict()
        for a in args:
            if isinstance(a, str):
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)
        return self

    def unvar(self, *args):
        for a in args:
            self.pop(a, None)
        return self

    def setdefaults(self, *args, **kwargs):
        args = [a for a in args if a not in self]
        kwargs = {k: v for k, v in kwargs.items() if k not in self}
        return self.var(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

    def new(self, *args, **kwargs):
        return Namespace({**self, **Namespace(*args, **kwargs)})

class Path(str):
    """
    Custom version of Python's pathlib.Path
    """
    @classmethod
    def env(cls, var):
        return Path(os.environ[var])

    def __init__(self, path):
        pass

    def __add__(self, subpath):
        return Path(str(self) + str(subpath))

    def __truediv__(self, subpath):
        return Path(os.path.join(str(self), str(subpath)))

    def __floordiv__(self, subpath):
        return (self / subpath).str

    def ls(self, show_hidden=True, dir_only=False, file_only=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if show_hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        if dir_only:
            return subdirs
        if file_only:
            return files
        return subdirs, files

    def lsdirs(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, dir_only=True)

    def lsfiles(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, file_only=True)

    def lslinks(self, show_hidden=True, exist=None):
        dirs, files = self.ls(show_hidden=show_hidden)
        return [x for x in dirs + files if x.islink() and (
            exist is None or not (exist ^ x.exists()))]

    def glob(self, glob_str):
        return [Path(p) for p in glob(self / glob_str, recursive=True)]

    def glob_rest(self, glob_str):
        return self.parent.glob(f'{self.name}{glob_str}')

    def re(self, re_pattern):
        """ Similar to .glob but uses regex pattern """
        subpatterns = lmap(re.compile, re_pattern.split('/'))
        matches = []
        dirs, files = self.ls()
        for pattern in subpatterns[:-1]:
            new_dirs, new_files = [], []
            for d in filter(lambda x: pattern.fullmatch(x.name), dirs):
                d_dirs, d_files = d.ls()
                new_dirs.extend(d_dirs)
                new_files.extend(d_files)
            dirs, files = new_dirs, new_files
        return sorted(filter(lambda x: subpatterns[-1].fullmatch(x.name), dirs + files))

    def recurse(self, dir_fn=None, file_fn=None):
        """ Recursively apply dir_fn and file_fn to all subdirs and files in directory """
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)

    def mk(self):
        os.makedirs(self, exist_ok=True)
        return self

    def mk_parent(self):
        self.parent.mk()
        return self
    dir_mk = mk_parent

    def rm(self):
        if self.isfile() or self.islink():
            os.remove(self)
        elif self.isdir():
            shutil.rmtree(self)
        return self

    def unlink(self):
        os.unlink(self)
        return self


    def mv(self, dest):
        shutil.move(self, dest)

    def mv_from(self, src):
        shutil.move(src, self)

    def cp(self, dest):
        shutil.copy(self, dest)

    def cp_from(self, src):
        shutil.copy(src, self)

    def link(self, target, force=False):
        if self.lexists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        return os.path.exists(self)

    def lexists(self):
        return os.path.lexists(self)

    def isfile(self):
        return os.path.isfile(self)

    def isdir(self):
        return os.path.isdir(self)

    def islink(self):
        return os.path.islink(self)

    def chdir(self):
        os.chdir(self)

    def rel(self, start=None):
        return Path(os.path.relpath(self, start=start))

    def clone(self):
        name = self.name
        match = re.search('__([0-9]+)$', name)
        if match is None:
            base = self + '__'
            i = 1
        else:
            initial = match.group(1)
            base = self[:-len(initial)]
            i = int(initial) + 1
        while True:
            path = Path(base + str(i))
            if not path.exists():
                return path
            i += 1

    @property
    def str(self):
        return str(self)
    @property
    def _(self):
        return self.str

    @property
    def real(self):
        return Path(os.path.realpath(os.path.expanduser(self)))
    @property
    def _real(self):
        return self.real

    @property
    def parent(self):
        path = os.path.dirname(self.rstrip('/'))
        if path == '':
            path = os.path.dirname(self.real.rstrip('/'))
        return Path(path)
    @property
    def _up(self):
        return self.parent

    @property
    def name(self):
        return Path(os.path.basename(self))
    @property
    def _name(self):
        return self.name

    @property
    def stem(self):
        return Path(os.path.splitext(self)[0])
    @property
    def _stem(self):
        return self.stem

    @property
    def basestem(self):
        new = self.stem
        while new != self:
            new, self = new.stem, new
        return new
    @property
    def _basestem(self):
        return self.basestem

    @property
    def ext(self):
        return Path(os.path.splitext(self)[1])
    @property
    def _ext(self):
        return self.ext

    extract = extract
    load_json = load_json
    save_json = save_json
    load_txt = load_sh = load_text
    save_txt = save_sh = save_text
    load_p = load_pickle
    save_p = save_pickle

    def save_bytes(self, bytes):
        with open(self, 'wb') as f:
            f.write(bytes)

    def load_csv(self, index_col=0, **kwargs):
        return pd.read_csv(self, index_col=index_col, **kwargs)

    def save_csv(self, df, float_format='%.5g', **kwargs):
        df.to_csv(self, float_format=float_format, **kwargs)

    def load_npy(self):
        return np.load(self, allow_pickle=True)

    def load_npz(self):
        return np.load(self, allow_pickle=True)

    def save_npy(self, obj):
        np.save(self, obj)

    def load_yaml(self):
        with open(self, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, obj):
        with open(self, 'w') as f:
            yaml.dump(convert_yaml(obj), f, default_flow_style=False, allow_unicode=True)

    def load_pth(self):
        return torch.load(self)

    def save_pth(self, obj):
        torch.save(obj, self)

    def load_feather(self):
        return pd.read_feather(self)
    load_ftr = load_feather

    def save_feather(self, df):
        return df.to_feather(self)
    save_ftr = save_feather

    def load_parquet(self):
        return pd.read_parquet(self)

    def save_parquet(self, df):
        return df.to_parquet(self)

    def load_pdf(self):
        """
        return: PdfReader object.
        Can use index and slice obj.pages for the pages, then call Path.save_pdf to save
        """
        from pdfrw import PdfReader
        return PdfReader(self)

    def save_pdf(self, pages):
        from pdfrw import PdfWriter
        writer = PdfWriter()
        writer.addpages(pages)
        writer.write(self)

    def load(self):
        return eval('self.load_%s' % self.ext[1:])()

    def save(self, obj):
        return eval('self.save_%s' % self.ext[1:])(obj)

    def replace_txt(self, replacements, dst=None):
        content = self.load_txt()
        for k, v in replacements.items():
            content = content.replace(k, v)
        (dst or self).save_txt(content)

    def update_dict(self, updates={}, vars=[], unvars=[], dst=None):
        d = self.load()
        for k in vars:
            d[k] = True
        for k in unvars:
            d.pop(k, None)
        d.update(updates)
        (dst or self).save(d)

    def torch_strip(self, dst):
        self.update_dict(unvars=['opt', 'step'], dst=dst)

    def wget(self, link):
        if self.isdir():
            return Path(wget(link, self))
        raise ValueError('Path %s needs to be a directory' % self)

    def replace(self, old, new=''):
        return Path(super().replace(old, new))

    def search(self, pattern):
        return re.search(pattern, self)

    def search_pattern(self, pattern):
        return self.search(pattern).group()

    def search_groups(self, pattern):
        return self.search(pattern).groups()

    def search_group(self, pattern):
        return self.search_groups(pattern)[0]

    def findall(self, pattern):
        return re.findall(pattern, self)