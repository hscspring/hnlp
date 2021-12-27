from hnlp.node import Node, N



def test_node():
    class A(Node):
        name: str = "A"
        def __post_init__(self):
            super().__init__()
            self.node = lambda x: "/".join([x, self.name])
    class B(Node):
        name: str = "B"
        def __post_init__(self):
            super().__init__()
            self.node = lambda x: "/".join([x, self.name])

    pipe = A() >> B()
    assert pipe.run("Run") == "Run/A/B"
    assert pipe("Run") == "Run/A"


def test_n():
    def load(x):
        return "/".join([x, "loaded"])
    def clean(x):
        return "/".join([x, "cleaned"])

    pipe = N(load) >> N(clean)
    assert pipe("Run") == "Run/loaded/cleaned"

    pipe = N() >> (filter, lambda x: x < 6) >> sum >> str >> N(clean)
    assert pipe(range(10)) == "15/cleaned"
