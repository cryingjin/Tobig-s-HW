import numpy as np
import math

class TwoLayerNet():
    """
    2 Layer Network를 만드려고 합니다.

    해당 네트워크는 아래의 구조를 따릅니다.

    input - Linear - ReLU - Linear - Softmax
     : X(n,d) -> H = X*W1+b1 -> A = Relu(H) -> S = A*W2+b2 -> p = softmax(s)
    L = -logliklihood(p)

    Softmax 결과는 입력 N개의 데이터에 대해 개별 클래스에 대한 확률입니다.
    즉 y의 클래스를 맞추는 범주형에 대한 과정임.
    """

    def __init__(self, X, input_size, hidden_size, output_size, std=1e-4):
         """
         네트워크에 필요한 가중치들을 initialization합니다.
         initialized by random values
         해당 가중치들은 self.params 라는 Dictionary에 담아둡니다.

         input_size: 데이터의 변수 개수 - D
         hidden_size: 히든 층의 H 개수 - H
         output_size: 클래스 개수 - C

         std, standard error 0.0001로 설정

         """
         # np.random.randn(x) : 가우시안 분포 하에서 x의 난수 matrix array 생성
         # 왜 bias 말고 weight에만 std곱해주는지는 모르겠음
         self.params = {}
         self.params["W1"] = std * np.random.randn(input_size, hidden_size)    #(d,h)
         self.params["b1"] = np.random.randn(hidden_size)    #(h) h개
         self.params["W2"] = std * np.random.randn(hidden_size, output_size) #(h,c)
         self.params["b2"] = np.random.randn(output_size)    #(c) c개, 마지막 output으로 나오는 calss의 개수 와 같음

    def forward(self, X, y=None):

        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        Linear - ReLU - Linear - Softmax - CrossEntropy Loss

        y가 주어지지 않으면 Softmax 결과 p와 Activation 결과 a를 return합니다. p와 a 모두 backward에서 미분할때 사용합니다.
        y가 주어지면 CrossEntropy Error를 return합니다.
        
        <참고>
        X(n,d) -> H = X*W1+b1 -> A = Relu(H) -> S = A*W2+b2 -> p = softmax(s)
        L = -loglikelihood(p)

        """

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # 여기에 p를 구하는 작업을 수행하세요.

        h = np.dot(X,W1) + b1     # H = X*W1 + b1
        a = np.maximum(0,h)            # A = Relu(H) -> ReLu함수는 max(0,x) 형태
        # a = np.maximum(0,h)       
        o = np.dot(a,W2) + b2     # o = s = A*W2 + b2, softmax전의 linear형태
        p = np.exp(o)/np.sum(np.exp(o),axis=1).reshape(-1,1)        # p는 ()*1 형태

        if y is None:
            return p, a          
        
        # 여기에 Loss를 구하는 작업을 수행하세요.
        # y는 범주형 변수, 다항분포로 추정, Loss function은 cross-entropy Loss 사용해야함
        # L = -loglikelihood(p)
        logloss = 0
        for i in range(p.shape[0]):
            index = y[i]
            Loss = np.log(p[i][index])
            logloss = logloss - Loss
    
        
        #Loss = -sum(p*np.log(p)+(1-p)*np.log(1-p))
        # 참고 : https://ai.stackexchange.com/questions/6622/cross-entropy-loss-function-causes-division-by-zero-error
        # 참고 : log loss -> 진혁 멘토가 도와줌

        #print('loss : ',Loss)

        return logloss



    def backward(self, X, y, learning_rate=1e-5):
        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        grads에는 Loss에 대한 W1, b1, W2, b2 미분 값이 기록됩니다.

        원래 backw 미분 결과를 return 하지만
        여기서는 Gradient Descent방식으로 가중치 갱신까지 합니다.

        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N = X.shape[0] # 데이터 개수
        grads = {}

        p, a = self.forward(X)        #forward(X) 는 loss값 scalar

        # 여기에 파라미터에 대한 미분을 저장하세요.

        dp = p
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if(j==y[i]):
                    dp[i][j]-=1      # SoftMax 미분 : dL/dS = P-T(label)
          # p-y
        da = np.heaviside(a,0)      # ReLu 미분 (heaviside) : a보다 작은 값 0, 같은 값 0, 큰 값 1
        """
                              0   if x1 < 0
        heaviside(x1, x2) =  x2   if x1 == 0
                              1   if x1 > 0

        """


        # grads : b2 -> w2 -> b1 -> w1
        grads["W2"] = np.dot(a.T,dp)      # dL/dW2 = dS/dW2 * dL/dS = A.T * dL/dS
        grads["b2"] = np.sum(dp,axis=0)      # dL/db2 = 1* dL/dS = P-Y
        
        grads["b1"] = np.sum(da*np.dot(dp,W2.T),axis=0)
        grads["W1"] = np.dot(X.T,da*np.dot(dp,W2.T))

        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]
        

    def accuracy(self, X, y):        # learning_rate조정해서 얻은 accuracy 값

        p, _ = self.forward(X)       # loss
        
        
        pre_p = np.argmax(p,axis=1)  # 최대화 시키는 p값

        return np.sum(pre_p==y)/pre_p.shape[0]
