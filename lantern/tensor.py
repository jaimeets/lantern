import numpy as np

class Tensor():
    def __init__(self, data, _children=[], requires_grad=False):
        self.data = np.array(data, dtype=np.float32)    
        self.requires_grad = requires_grad
        self._children = _children
        self._backward = lambda : None
        self.grad = np.zeros(self.data.shape, dtype=np.float32)
        self._mygrad = np.zeros(self.data.shape, dtype=np.float32)

#requires_grad
#ops with scalars
#node accumulation
    def __add__(self, other):
        if not isinstance(other, (Tensor)):
            other = Tensor(other)
        out = Tensor(self.data + other.data)
        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self._mygrad = out._mygrad
                    self.grad += self._mygrad
                if other.requires_grad:
                    other._mygrad = out._mygrad
                    other.grad += other._mygrad
            out._backward = _backward    
        return out
    
    def __mul__(self, other):
        if not isinstance(other, (Tensor)):
            other = Tensor(other)
        out = Tensor(self.data * other.data)
        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self._mygrad = other.data * out._mygrad
                    self.grad += self._mygrad
                if other.requires_grad:
                    other._mygrad = self.data * out._mygrad
                    other.grad += other._mygrad
            out._backward = _backward    
        return out
 
    def __pow__(self, other):
        if not isinstance(other, (Tensor)):
            other = Tensor(other)
        out = Tensor(self.data ** other.data)
        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self._mygrad = (other.data * (self.data ** (other.data-1))) * out._mygrad
                    self.grad += self._mygrad
                if other.requires_grad:
                    other._mygrad = ((self.data ** other.data) * np.log(self.data)) * out._mygrad
                    other.grad += other._mygrad
            out._backward = _backward    
        return out    

    def matmul(self, other):
        if not isinstance(other, (Tensor)):
            raise TypeError("matmul(): argument 'input' (position 2) must be Tensor")
        try:
            out_err = np.matmul(self.data, other.data)
        except ValueError:
            raise ValueError(f'shapes {self.data.shape} and {other.data.shape} not aligned')
        out = Tensor(np.matmul(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self._mygrad = np.matmul(out._mygrad, other.data.T)
                    self.grad += self._mygrad
                if other.requires_grad:
                    other._mygrad = np.matmul(self.data.T, out._mygrad)
                    other.grad += other._mygrad
            out._backward = _backward
        return out

    def relu(self): 
        data = np.where(self.data > 0, self.data, 0.0)
        out = Tensor(data, [self], requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self._mygrad = np.where(out.data > 0, 1.0, 0.0) * out._mygrad
                self.grad += self._mygrad
        out._backward = _backward
        return out

    def eye(data, requires_grad=False):
        return Tensor(np.eye(data), requires_grad=requires_grad)

    def __neg__(self):
        return self.__mul__(-1)
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        return self.__add__(-other)
    def __truediv__(self, other):
        return self * (other ** -1)
    def __rtruediv__(self, other):
        return (self ** -1) * other
    def __rsub__(self, other):
        data = -self
        return data.__add__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rpow__(self, other):
        other = Tensor(other)
        return other.__pow__(self)

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

    def backward(self):
        order = []
        visited = []
        def topological_sort(V):
            if V not in visited:
                visited.append(V)
                for child in V._children:           
                    topological_sort(child)
            order.append(V)
        topological_sort(self)
        order.reverse()
        if self.requires_grad:
            self.grad = np.ones(self.data.shape, dtype=np.float32)
            self._mygrad = np.ones(self.data.shape, dtype=np.float32)
            for V in order:
                V._backward()  

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape, dtype=np.float32)
        self._mygrad = np.zeros(self.data.shape, dtype=np.float32)

