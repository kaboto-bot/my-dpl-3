import numpy as np

class Variable():
    def __init__(self, pData):
        # ndarrayだけを扱う
        if pData is not None:
            if not isinstance(pData, np.ndarray):
                raise TypeError('{} is not supported\n※ numpy.ndarrayタイプ以外はサポートしません'.format(type(pData)))
        self.mData = pData
        self.mGenerateFunc = None
        self.mGrad = None

    def SetGenerateFunc(self, pGenerateFunc):
        self.mGenerateFunc = pGenerateFunc
    
    def SetGrad(self, pGrad):
        self.mGrad = pGrad
    
    # 再起的にBackwardする
    def Backward(self):
        # Gradが設定されていない場合1で初期化する
        if self.mGrad == None:
            self.mGrad = np.ones_like(self.mData)
        funclist = [self.mGenerateFunc]
        while (funclist):
            func = funclist.pop()
            output_grad = func.mOutput.mGrad
            input_grad = func.CallBackward(output_grad)
            func.mInputVar.SetGrad(input_grad)
            if None is not func.mInputVar.mGenerateFunc:
                funclist.append(func.mInputVar.mGenerateFunc)

    def PrintLog(self):
        print("data = {}".format(self.mData))
        print("grad = {}".format(self.mGrad))
        print("generate_func = {}".format(self.mGenerateFunc))

class Function():
    def __call__(self, pInputVar):
        y = self.CallForward(pInputVar)
        self.mOutput = Variable(as_array(y))
        self.mOutput.SetGenerateFunc(self)
        return self.mOutput
    
    # Override前提
    def forward(self, pInputData):
        raise NotImplementedError
    
    # Override前提
    def backward(self, pInputData):
        raise NotImplementedError
    
    def CallBackward(self, pParentGrad):
        grad = self.backward(self.mInputVar.mData)
        return grad * pParentGrad
    
    def CallForward(self, pInputVar):
        self.mInputVar = pInputVar
        return self.forward(pInputVar.mData)


class Square(Function):
    def forward(self, pInputData):
        return pInputData ** 2

    def backward(self, pInputData):
        grad = pInputData * 2
        return grad


def square(pInput):
    return Square() (pInput)

class Exp(Function):
    def forward(self, pInputData):
        y = np.exp(pInputData)
        return y
    
    def backward(self, pInputData):
        grad = np.exp(pInputData)
        return grad

# np.scalarをnp.ndarray[0dim]に変換する
def as_array(pX):
    if np.isscalar(pX):
        return np.array(pX)
    return x

x = Variable(np.array(2))
x.PrintLog()
y = square(x)
y.PrintLog()

y.Backward()
x.PrintLog()
y.PrintLog()

