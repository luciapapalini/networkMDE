import unittest
from networkMDE.classiter import clist, cset, cdict


class dummyClass:
    def __init__(self, a_normal_attribute):
        self.a_normal_attribute = a_normal_attribute

    @property
    def attribute(self):
        return self.a_normal_attribute

    def method(self):
        return self.a_normal_attribute + 3

    def __hash__(self):
        return hash(self.a_normal_attribute)

    def __eq__(self, other):
        return self.a_normal_attribute == other.a_normal_attribute


class citerTest(unittest.TestCase):
    def setUp(self):

        self.list_of_attributes = [1, 2, 3, 3, 3, 5]
        self.clist_of_attributes = clist(self.list_of_attributes)
        self.clist_of_list_set = clist(list(set(self.list_of_attributes)))

        self.clist = clist()
        self.cset = cset()
        self.cdict = cdict()

        self.obj_type = type(dummyClass(42))

    def test_init_type(self):

        msg = "A void citer should have type None"
        self.assertIsNone(self.clist.type, msg)
        self.assertIsNone(self.cset.type, msg)
        self.assertIsNone(self.cdict.type, msg)

    def test_addition(self):

        for attr in self.list_of_attributes:
            self.clist += dummyClass(attr)
            self.cset += dummyClass(attr)
            self.cdict += {attr: dummyClass(attr)}

        self.assertTrue(
            self.clist.type == self.obj_type,
            f"type should be {self.obj_type}, not {self.clist.type}",
        )
        self.assertTrue(
            self.cdict.type == self.obj_type,
            f"type should be {self.obj_type}, not {self.cdict.type}",
        )
        self.assertTrue(
            self.cset.type == self.obj_type,
            f"type should be {self.obj_type}, not {self.cset.type}",
        )

    def test_getattr(self):

        self.clist = clist([dummyClass(_) for _ in self.list_of_attributes])
        self.cset = cset({dummyClass(_) for _ in self.list_of_attributes})
        self.cdict = cdict({_: dummyClass(_) for _ in self.list_of_attributes})

        self.assertTrue(
            self.clist.attribute == self.clist_of_attributes,
            f"clist attribute should be {self.clist_of_attributes} not {self.clist.attribute}",
        )

        # Verifies that the __hash__ and __eq__ methods make elements indistinguishable
        self.assertTrue(
            self.cset.attribute == self.clist_of_list_set,
            f"cset attribute should be {self.clist_of_list_set} not {self.cset.attribute}",
        )

        self.assertTrue(
            self.cdict.attribute == self.clist_of_list_set,
            f"cdict attribute should be {self.clist_of_list_set} not {self.cdict.attribute}",
        )

    def test_method(self):

        self.clist = clist([dummyClass(_) for _ in self.list_of_attributes])
        self.cset = cset({dummyClass(_) for _ in self.list_of_attributes})
        self.cdict = cdict({_: dummyClass(_) for _ in self.list_of_attributes})

        self.assertEqual(
            self.clist.method(),
            clist([_ + 3 for _ in self.list_of_attributes]),
            f"method calling: {self.clist.method()} !=  {clist([_+3 for _ in self.list_of_attributes])}",
        )

        self.assertEqual(
            self.cset.method(),
            clist([_ + 3 for _ in self.clist_of_list_set]),
            f"method calling: {self.cset.method()} !=  {clist([_+3 for _ in self.clist_of_list_set])}",
        )

        self.assertEqual(
            self.cdict.method(),
            clist([_ + 3 for _ in self.clist_of_list_set]),
            f"method calling: {self.cdict.method()} !=  {clist([_+3 for _ in self.clist_of_list_set])}",
        )


if __name__ == "__main__":

    unittest.main()
