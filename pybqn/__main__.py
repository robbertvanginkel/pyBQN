import pickle
import pickletools

class PyBQNVM:
    Foo : str = 'fp'


def parse_bytecode(bytecode: str):
    pass

if __name__ == "__main__":
    x = PyBQNVM()
    x.Foo = 'bar'
    picklestring = pickle.dumps(x, protocol=1)
    print(picklestring)
    print(pickletools.dis(picklestring))

