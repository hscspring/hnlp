from dataclasses import dataclass
from hnlp.register import Register


def test_register():
    @Register.register
    @dataclass
    class MyClassSomething:
        pass

    ins1 = Register.get("MyClassSomething")()
    ins2 = Register.get("my-class-something")()

    ins0 = MyClassSomething()

    assert ins1 == ins2 == ins0
