"""Utility module to extend the getattr() method over iterables of class instances.

author: djanloo
date: 3 dic 2021

I am not aware of any built-in neither third-party way to do this,
so I did it by myself.

This module basically let this

    A = citer([inst1, inst2, inst3])
    print(A.attr)

    #prints [int1.attr, inst2.attr, inst3.attr]

become possible, while conserving (hopefully) most compatibility
with other methods
"""


class citer:

    citer_name = "citer_template"

    def __init__(self, objs=None):
        self.objs = objs if objs is not None else self.empty

        self._type = None
        if hasattr(objs, "__iter__"):
            for obj in objs:
                self.type = type(obj)

        if objs:
            for obj in objs:
                self.type = type(obj)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, set_type):
        if self._type is None or set_type == self._type:
            self._type = set_type
        else:
            raise TypeError(
                f"clist type is already set to {self._type} (requested switch to {set_type})"
            )

    def __setitem__(self, key, value):
        self.objs[key] = value

    def __getattr__(self, attr_name):
        return clist([getattr(obj, attr_name) for obj in self.objs])

    def __str__(self):
        desc = f"{self.__class__.citer_name} of {str(self.type)} ["
        for obj in self.objs:
            desc += f" {str(obj)} \n"
        desc += "]"
        return desc

    def __iter__(self):
        return iter(self.objs)

    def __next__(self):
        return next(self.objs)

    def __len__(self):
        return len(self.objs)

    def __eq__(self, other):
        return self.objs == other.objs

    def __call__(self, *args, **kwargs):
        return clist([obj(*args, **kwargs) for obj in self.objs])


class clist(citer):

    citer_name = "clist"

    def __init__(self, objs=None):
        self.empty = list()
        super().__init__(objs)

    def __iadd__(self, element):
        self.type = type(element)
        self.objs.append(element)
        return self

    def __getitem__(self, index):
        try:
            item = self.objs[index]
        except KeyError:
            raise KeyError(f"Requested element {index} of {self.objs}")
        return item

    def __list__(self):
        return self.objs


class cdict(citer):

    citer_name = "cdict"

    def __init__(self, objs=None):
        self.empty = dict()
        super().__init__(objs)

    def __getattr__(self, attr_name):
        return clist([getattr(obj, attr_name) for obj in self.objs.values()])

    def get(self, key, default=None):
        return self.objs.get(key, default)

    def __iadd__(self, element):
        for key, value in element.items():
            self.type = type(value)
            self.objs[key] = value
        return self

    def __getitem__(self, index):
        try:
            item = self.objs[index]
        except KeyError:
            raise KeyError(f"Requested element {index} of {self.objs}")
        return item

    def __str__(self):
        desc = f"{self.__class__.citer_name} of {str(self.type)} ["
        for key, value in self.objs.items():
            desc += f" {str(key)}:{str(value)} "
        desc += "]"
        return desc

    def __iter__(self):
        return iter(self.objs.values())

    def __next__(self):
        return next(self.objs.values())

    def __call__(self, *args, **kwargs):
        return clist([obj(*args, **kwargs) for obj in self.objs.values()])


class cset(citer):

    citer_name = "cset"

    def __init__(self, objs=None):
        self.empty = set()
        objs = set(objs) if objs is not None else objs
        super().__init__(objs)

    def __iadd__(self, element):
        self.objs.add(element)
        self.type = type(element)
        return self
