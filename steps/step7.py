import numpy as np

class Variable():
    def __init__(self, pData):
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
            input_grad = func.backward(output_grad)
            func.mInput.SetGrad(input_grad)
            if None is not func.mInput.mGenerateFunc:
                funclist.append(func.mInput.mGenerateFunc)

    def PrintLog(self):
        print("data = {}".format(self.mData))
        print("grad = {}".format(self.mGrad))
        print("generate_func = {}".format(self.mGenerateFunc))

class Function():
    def __call__(self, pInput):
        x = pInput.mData
        self.mInput = pInput
        y = self.forward(x)
        self.mOutput = Variable(y)
        self.mOutput.SetGenerateFunc(self)
        return self.mOutput
    
    def forward(self, pData):
        raise NotImplementedError
    
    def backward(self, pBackGrad):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, pData):
        return pData ** 2

    def backward(self, pBackGrad):
        grad = self.mInput.mData * 2
        return grad * pBackGrad


def square(pInput):
    return Square() (pInput)

x = Variable(np.array(2))
x.PrintLog()
y = square(x)
y.PrintLog()

y.Backward()
x.PrintLog()
y.PrintLog()

