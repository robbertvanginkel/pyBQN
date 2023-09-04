from pybqn import VM

if __name__ == "__main__":
    vm = VM()
    result = vm.run('<⟜\'a\'⊸/ "Big Questions Notation"')
    print(vm.format(result))
