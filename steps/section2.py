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
        if self.mGrad is None:
            self.mGrad = pGrad
        else :
            self.mGrad += pGrad
    
    # 再起的にBackwardする
    def Backward(self):
        # Gradが設定されていない場合1で初期化する
        if self.mGrad is None:
            self.mGrad = np.ones_like(self.mData)
        funclist = [self.mGenerateFunc]
        while (funclist):
            func = funclist.pop()
            output_grads = [output_var.mGrad for output_var in func.mOutputs]
            input_grads = func.CallBackward(output_grads)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            for input_var, input_grad in zip(func.mInputs, input_grads):
                input_var.SetGrad(input_grad)
                if None is not input_var.mGenerateFunc:
                    funclist.append(input_var.mGenerateFunc)

    def PrintLog(self):
        print(f"data = {self.mData}")
        print(f"grad = {self.mGrad}")
        print(f"generate_func = {self.mGenerateFunc}")

class Function():
    def __call__(self, *pInputVars):
        # フォーワード前処理: 入力を記録
        self.mInputs = pInputVars

        # フォーワード処理 入力もタプル、出力はタプル
        output_vars = self.CallForward(pInputVars)

        # フォーワード後処理: 出力した値を加工
        self.mOutputs = output_vars
        for output_var in output_vars:
            output_var.SetGenerateFunc(self)

        # 出力が１つだったら１つだけ返す その他はリストで返す
        return output_vars[0] if len(output_vars) == 1 else output_vars
    
    # Override前提
    def forward(self, *pInputDatas):
        raise NotImplementedError
    
    # Override前提
    def backward(self, *pInputDatas):
        raise NotImplementedError
    
    # 入力はタプル
    def CallBackward(self, pOutputGrads):
        input_datas = [input_var.mData for input_var in self.mInputs]
        grads = self.backward(*input_datas)
        if not isinstance(grads, tuple):
            grads = (grads,)
        input_grads = grads * sum(pOutputGrads)
        return tuple(input_grads)
    
    # 入力はタプル
    def CallForward(self, pInputVars):
        # Variableから値だけを取り出したリストを作成
        input_datas = [input_var.mData for input_var in pInputVars]
        # リストをアンパッキングして複数引数にしてforwardに渡す
        y = self.forward(*input_datas)
        output_datas = y
        if not isinstance(y, tuple):
            output_datas = (y,)
        output_vars = [Variable(as_array(y)) for y in output_datas]
        return output_vars

class Add(Function):
    def forward(self, pX0, pX1):
        y = pX0 + pX1
        return y
    
    def backward(self, *pInputDates):
        dx0, dx1 = 1, 1
        return (dx0, dx1)

def add(pX0, pX1):
    return Add() (pX0, pX1)

# np.scalarをnp.ndarray[0dim]に変換する
def as_array(pX):
    if np.isscalar(pX):
        return np.array(pX)
    return pX



x1 = Variable(np.array(2))
x2 = Variable(np.array(3))
print("hello")
y = add(x1, x2)
y.Backward()
y.PrintLog()
x1.PrintLog()
x2.PrintLog()

