import numpy as np

class Variable():
    """
    A class representing a variable with automatic differentiation capabilities.

    Attributes:
        mData (np.ndarray): The data of the variable.
        mGenerateFunc (Function): The function that generated this variable.
        mGrad (np.ndarray): The gradient of the variable.
    """
    def __init__(self, pData):
        """
        Initialize a Variable instance.

        Args:
            pData (np.ndarray): The data to be stored in the variable. Must be a numpy ndarray.
        """
        # ndarrayだけを扱う
        if pData is not None:
            if not isinstance(pData, np.ndarray):
                raise TypeError('{} is not supported\n※ numpy.ndarrayタイプ以外はサポートしません'.format(type(pData)))
        self.mData = pData
        self.mGenerateFunc = None
        self.mGrad = None
    
    def SetGenerateFunc(self, pGenerateFunc):
        """
        Set the function that generated this variable.

        Args:
            pGenerateFunc (Function): The function to be set as the generator.
        """
        self.mGenerateFunc = pGenerateFunc
    
    def SetGrad(self, pGrad):
        """
        Set the gradient of the variable. If the gradient already exists, add the new gradient to it.

        Args:
            pGrad (np.ndarray): The gradient to be set.
        """
        if self.mGrad is None:
            self.mGrad = pGrad
        else :
            self.mGrad += pGrad
    
    def ResetGrad(self):
        """
        Reset the gradient of the variable.
        """
        self.mGrad = None
    
    # 再起的にBackwardする
    def Backward(self):
        """
        Perform backpropagation to compute the gradients of all variables.
        """
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
    """
    A base class for all functions used in the computation graph.

    Methods:
        __call__(self, *pInputVars): Calls the forward method and processes the output.
        forward(self, *pInputDatas): The forward computation (to be overridden).
        backward(self, *pInputDatas): The backward computation (to be overridden).
        CallBackward(self, pOutputGrads): Calls the backward method with the gradients of the outputs.
        CallForward(self, pInputVars): Calls the forward method with the inputs.
    """

    def __call__(self, *pInputVars):
        """
        Perform the forward pass and setup for backpropagation.

        Args:
            *pInputVars (Variable): Input variables.

        Returns:
            Variable or (tuple of Variable): Output variable(s).
        """

        # フォーワード前処理: 入力を記録
        self.mInputs = pInputVars

        # フォーワード処理 入力もタプル、出力もタプル
        output_vars = self.CallForward(pInputVars)

        # フォーワード後処理: 出力した値を加工
        self.mOutputs = output_vars
        for output_var in output_vars:
            output_var.SetGenerateFunc(self)

        # 出力が１つだったら１つだけ返す その他はタプルで返す
        return output_vars[0] if len(output_vars) == 1 else output_vars
    
    # Override前提
    def forward(self, *pInputDatas):
        """
        The forward computation. To be overridden by subclasses.

        Args:
            *pInputDatas (np.ndarray): Input data(s).
        """
        raise NotImplementedError
    
    # Override前提
    def backward(self, *pInputDatas):
        """
        The backward computation. To be overridden by subclasses.

        Args:
            *pInputDatas (np.ndarray): Input data(s). same of forward arg pInputDatas.
        """
        raise NotImplementedError
    
    # 入力はタプル
    def CallBackward(self, pOutputGrads):
        """
        Calls the backward method with the gradients of the outputs.

        Args:
            pOutputGrads (tuple of np.ndarray): Gradients of the output variables.

        Returns:
            tuple of np.ndarray: Gradients of the input variables.
        """
        input_datas = [input_var.mData for input_var in self.mInputs]
        grads = self.backward(*input_datas)
        if not isinstance(grads, tuple):
            grads = (grads,)
        input_grads = grads * sum(pOutputGrads)
        return tuple(input_grads)
    
    # 入力はタプル、出力もタプル
    def CallForward(self, pInputVars):
        """
        Calls the forward method with the input variables.

        Args:
            pInputVars (tuple of Variable): Input variables.

        Returns:
            tuple of Variable: Output variables.
        """
        # Variableから値だけを取り出したリストを作成
        input_datas = [input_var.mData for input_var in pInputVars]
        # タプルをアンパッキングして複数引数にしてforwardに渡す
        y = self.forward(*input_datas)
        output_datas = y
        if not isinstance(y, tuple):
            output_datas = (y,)
        output_vars = tuple([Variable(as_array(y)) for y in output_datas])
        return output_vars

class Add(Function):
    def forward(self, pX0, pX1):
        y = pX0 + pX1
        return y
    
    def backward(self, *pInputDates):
        dx0, dx1 = 1, 1
        return (dx0, dx1)

def add(pX0, pX1):
    """
    Add two variables using the Add function.

    Args:
        pX0 (Variable): The first input variable.
        pX1 (Variable): The second input variable.

    Returns:
        Variable: The output variable representing the sum of the inputs.
    """
    return Add() (pX0, pX1)

def as_array(pX):
    """
    Convert np.scalar to np.ndarray(0,)

    Args:
        pInputVars (np.ndarray or np.scalar): Input Data.

    Return:
        np.ndarray : 
    """
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

