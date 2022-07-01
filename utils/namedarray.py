from collections.abc import Mapping
import copy
import dataclasses
import itertools
import logging
import numpy as np
import torch

logger = logging.getLogger("namedarray")
logger.setLevel(logging.INFO)


def _is_dataclass_instance(obj):
    # this is used for checking the same data structure
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_namedarray_instance(obj):
    flag = _is_dataclass_instance(obj) and hasattr(obj, '_fields')
    flag = flag and hasattr(obj, 'items') and issubclass(type(obj), Mapping)
    return flag and not isinstance(obj, type)


def _match_structure(template, value):
    if not (_is_dataclass_instance(value) and  # Check for matching structure.
            getattr(value, "_fields", None) == template._fields):
        if not _is_dataclass_instance(value):
            # Repeat value for each but respect any None.
            value = tuple(None if s is None or k == 'metadata' else value
                          for k, s in template.items())
        else:
            raise ValueError(
                'namedarray - operation with a different data structure')
    else:
        if hasattr(value, "metadata") and value.metadata is not None:
            logger.debug(
                "namedarray - metadata of the second operand is not None. It'll be ignored."
            )
    return value


def namedarray_op(cmd, is_iop):

    def fn(self, value):
        value = _match_structure(self, value)
        try:
            xs = []
            for j, (s, v) in enumerate(zip(self, value)):
                # since the metadata of the second operand is None,
                # we can conduct this operation safely
                if s is not None and v is not None:
                    if is_iop:
                        exec(f"s {cmd} v")
                    else:
                        exec(f"xs.append(s {cmd} v)")
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(
                f"{type(e).__name__} occured in {self.__class__.__name__}"
                " at field "
                f"'{self._fields[j]}': {e}") from e
        # NOTE: metadata is not changed implicitly
        return self if is_iop else type(self)(*xs)

    return fn


def namedarray(cls, *args, **kwargs):
    """A class decorator modified from the `namedarraytuple` class in rlpyt repo,
    referring to
    https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/collections.py#L16.

    A decorated namedarray implies a dataclasses.dataclass. It further supports
    dict-like unpacking and string indexing, and exposes integer slicing reads
    and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    Note that namedarray supports recursive definition,
    i.e., the elements of which could also be namedarray.

    Update 20220522:
    namedarray supports in-class definition of metadata. Metadata does not
    share indexing (__getitem__) behavior with data entries, and thus can have
    different shapes and data types.
    Metadata will not be changed for inplace operation, and will be set to None
    for other operations (e.g., recursive_apply, recursive_aggregate, arithmetics).
    It should be always explicitly set via __setattr__.

    Example:
    >>> @namedarray
    ... class Point:
    ...     x: np.ndarray
    ...     y: np.ndarray
    ...
    >>> p=Point(np.array([1,2]), np.array([3,4]))
    >>> p
    Point(x=array([1, 2]), y=array([3, 4]))
    >>> p[:-1]
    Point(x=array([1]), y=array([3]))
    >>> p[0]
    Point(x=1, y=3)
    >>> p.x
    array([1, 2])
    >>> p['y']
    array([3, 4])
    >>> p[0] = 0
    >>> p
    Point(x=array([0, 2]), y=array([0, 4]))
    >>> p[0] = Point(5, 5)
    >>> p
    Point(x=array([5, 2]), y=array([5, 4]))
    >>> 'x' in p
    True
    >>> list(p.keys())
    ['x', 'y']
    >>> list(p.values())
    [array([5, 2]), array([5, 4])]
    >>> for k, v in p.items():
    ...     print(k, v)
    ...
    x [5 2]
    y [5 4]
    >>> def foo(x, y):
    ...     print(x, y)
    ...
    >>> foo(**p)
    [5 2] [5 4]
    """
    data_cls = dataclasses.dataclass(cls, *args, **kwargs)

    metadata_cls = data_cls.MetaData if hasattr(data_cls, "MetaData") else None
    typename = data_cls.__class__.__name__

    data_fields = tuple(k for k in data_cls.__dataclass_fields__.keys()
                        if k != 'metadata')

    metadata_fields = None
    if metadata_cls is not None:
        metadata_fields = data_cls.MetaData.__dataclass_fields__.keys()

    @property
    def _fields(self):
        """Dataclass fields of data entries.

        Returns:
            list: datalcass fields.
        """
        return data_fields

    @property
    def _metadata_fields(self):
        """Dataclass fields of metadata.

        Returns:
            list: metadata fields or None if MetaData
                is not defined in the namedarray.
        """
        return metadata_fields

    def __getattr__(self, loc):
        """Get attributes in a namedarray.

        Metadata can be directly accessed, i.e.,
            policy_version = sample.policy_version
        instead of
            policy_version = sample.metadata.policy_version.

        Args:
            loc (str): attribute name to be accessed.

        Returns:
            Any: namedarray attribute.
        """
        if metadata_fields is not None and loc in metadata_fields:  # metadata is defined
            return None if self.metadata is None else getattr(
                self.metadata, loc)
        return data_cls.__getattribute__(self, loc)

    def __setattr__(self, loc, value):
        """Set attributes in a namedarray.

        Metadata can be directly accessed, i.e.,
            sample.policy_version = 1
        instead of
            sample.metadata.policy_version = 1.

        Args:
            loc (str): attribute name to be changed.
            value (Any): target value.
        """
        if metadata_fields is not None and loc in metadata_fields:  # metadata is defined
            if self.metadata is None:  # metadata is not instantiated
                self.metadata = data_cls.MetaData(**{
                    loc: value,
                    **{k: None
                       for k in metadata_fields if k != loc}
                })
            else:
                setattr(self.metadata, loc, value)
        self.__dict__[loc] = value

    def __getitem__(self, loc):
        """Get item like a numpy array or a dict.
        
        If the index is string, return getattr(self, index).
        If the index is integer/slice, return a new dataclass instance containing
        the selected index or slice from each field.

        Args:
            loc (str or slice): Key or indices to get.

        Raises:
            Exception: To locate in which field the error occurs.

        Returns:
            Any: An element of the dataclass or a new dataclass
                object composed of the subarrays.
        """
        if isinstance(loc, str):
            # str indexing like in dict
            return getattr(self, loc)
        else:
            try:
                # the metadata of a sliced namedarray is None
                return type(self)(
                    *(None if s is None else s if k == 'metadata' else s[loc]
                      for k, s in self.items()))
            except IndexError as e:
                for j, s in enumerate(self):
                    if s is None:
                        continue
                    try:
                        _ = s[loc]
                    except IndexError:
                        raise Exception(
                            f"Occured in {self.__class__} at field "
                            f"'{self._fields[j]}'.") from e

    def __setitem__(self, loc, value):
        """Set item like a numpy array or a dict.

        If input value is the same dataclass type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields. Ignore fields that are both None.

        Args:
            loc (str or slice): Key or indices to set.
            value (Any): A dataclass instance with the same structure
                or elements of the dataclass object.

        Raises:
            Exception: To locate in which field the error occurs.
        """
        if isinstance(loc, str):
            setattr(self, loc, value)
        else:
            value = _match_structure(self, value)
            try:
                for j, (s, v) in enumerate(zip(self, value)):
                    if s is not None and v is not None:
                        s[loc] = v
            except (ValueError, IndexError, TypeError) as e:
                raise Exception(
                    f"{type(e).__name__} occured in {self.__class__.__name__}"
                    " at field "
                    f"'{self._fields[j]}': {e}") from e

    def __iter__(self):
        """Generator function to iterate over a namedarray.

        Only data entry is iterated (like a namedtuple) and metadata is omitted.
        """

        def gen():
            for k in self._fields:
                yield getattr(self, k)

        return gen()

    def __contains__(self, key):
        """Checks presence of a field name in data entries.

        Args:
            key (str): The queried field name.

        Returns:
            bool: Query result.
        """
        return key in self._fields

    def values(self):
        """Iterator over all data entry values.
        """
        for v in self:
            yield v

    def keys(self):
        """Iterate over all data entry keys.
        """
        for k in self._fields:
            yield k

    def items(self):
        """Iterate over ordered (field_name, value) pairs of data entries.

        Yields:
            tuple[str,Any]: (field_name, value) pairs
        """
        for k, v in zip(self._fields, self):
            yield k, v

    def to_dict(self):
        """Convert a namedarray into a (nested) dict.

        Metadata is omited.

        Returns:
            dict: converted dict.
        """
        result = {}
        for k, v in self.items():
            if is_namedarray_instance(v):
                result[k] = v.to_dict()
            elif v is None:
                result[k] = None
            else:
                result[k] = v
        return result

    @property
    def shape(self):
        return recursive_apply(self, lambda x: x.shape).to_dict()

    def size(self):
        return self.shape

    methods = [
        __getattr__, __setattr__, __getitem__, __setitem__, __iter__,
        __contains__, values, keys, items
    ]

    for method in methods:
        method.__qualname__ = f'{typename}.{method.__name__}'

    ops = {
        '__add__': namedarray_op('+', is_iop=False),
        '__sub__': namedarray_op('-', is_iop=False),
        '__mul__': namedarray_op('*', is_iop=False),
        '__truediv__': namedarray_op('/', is_iop=False),
    }
    iops = {
        '__iadd__': namedarray_op('+=', is_iop=True),
        '__isub__': namedarray_op('-=', is_iop=True),
        '__imul__': namedarray_op('*=', is_iop=True),
        '__itruediv__': namedarray_op('/=', is_iop=True),
    }
    for name, op in itertools.chain(ops.items(), iops.items()):
        op.__qualname__ = f'{typename}.{name}'

    arg_list = repr(list(data_cls.__dataclass_fields__.keys())).replace(
        "'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__iter__': __iter__,
        '_fields': _fields,
        '_metadata_fields': _metadata_fields,
        '__getattr__': __getattr__,
        '__setattr__': __setattr__,
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'items': items,
        'keys': keys,
        'values': values,
        'to_dict': to_dict,
        'shape': shape,
        'size': size,
    }
    class_namespace = {**class_namespace, **ops, **iops}
    for k, v in class_namespace.items():
        assert (metadata_fields is None
                or k not in metadata_fields) and k not in data_fields
        setattr(data_cls, k, v)

    Mapping.register(data_cls)
    return data_cls


def array_like(x, default_value=0):
    """Instantiate a new namedarray with the same data as x.

    Metadata is omitted.

    Args:
        x (namedarray): the template.
        default_value (int, optional): default value of the new array. Defaults to 0.

    Returns:
        namedarray: the new namedarray.
    """
    if is_namedarray_instance(x):
        return type(x)(*[array_like(xx, default_value) for xx in x])
    else:
        if isinstance(x, np.ndarray):
            data = np.zeros_like(x)
        else:
            assert isinstance(x, torch.Tensor), (
                'Currently, namedarray only supports'
                f' torch.Tensor and numpy.array (input is {type(x)})')
            data = torch.zeros_like(x)
        if default_value != 0:
            data[:] = default_value
        return data


def __array_filter_none(xs):
    is_not_nones = [x is not None for x in xs]
    if all(is_not_nones) or all(x is None for x in xs):
        return
    else:
        example_x = xs[is_not_nones.index(True)]
        for i, x in enumerate(xs):
            xs[i] = array_like(example_x) if x is None else x


def recursive_aggregate(xs, aggregate_fn):
    """Recursively aggregate a list of namedarray instances.

    Basically recursive stacking or concatenating.
    Metadata is omitted.

    Args:
        xs (List[Any]): A list of namedarrays or
            appropriate aggregation targets (e.g. numpy.ndarray).
        aggregate_fn (function): The aggregation function to be applied.

    Returns:
        Any: The aggregated result with the same data type of elements in xs.
    """
    __array_filter_none(xs)
    assert all([type(x) == type(xs[0]) for x in xs]), ([type(x)
                                                        for x in xs], xs)

    if is_namedarray_instance(xs[0]):
        return type(xs[0])(*[
            recursive_aggregate([x[k] for x in xs], aggregate_fn)
            for k in xs[0].keys()
        ])
    elif xs[0] is None:
        return None
    else:
        return aggregate_fn(xs)


def recursive_apply(x, fn):
    """Recursively apply a function to a namedarray x.

    Args:
        x (Any): The instance of a namedarray subclass
            or an appropriate target to apply fn.
        fn (function): The function to be applied.
    """
    if is_namedarray_instance(x):
        return type(x)(*[recursive_apply(v, fn) for v in x.values()])
    elif x is None:
        return None
    else:
        return fn(x)
