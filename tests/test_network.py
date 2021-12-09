import unittest
from networkMDE.classiter import cdict, cset, clist
from networkMDE.network import Node, uniLink, uniNetwork

class testNodeLinks(unittest.TestCase):

    def setUp(self):
        self.nodes = cdict({i:Node(i) for i in range(5)})
        self.links = cset()
        self.link_sparse = [[1,2,0.1], [2,1, 0.1], [1,3, 0.5], [3,4, 1.], [4,3, 1.]]
        for i,j,d in self.link_sparse:
            self.links += self.nodes[i].connect(self.nodes[j], d)
        print(self.links)

    def test_links(self):
        self.assertEqual(len(self.links), 3, f"uniLink(i,j) should be equivalent to uniLink(j,i): {self.links}, {self.link_sparse}")

if __name__ == "__main__":
    unittest.main()