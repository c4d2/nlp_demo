# (1)onehot..py实现独热编码，具体功能是根据输入数据输出指定数字的独热编码，输入数据共3列，每一列表示不同的特征及取值范围，程序输出结果为数字0、1、3的独热编码。
from sklearn import preprocessing #导入预处理模型
enc = preprocessing.OneHotEncoder()#调用独热编码
enc.fit([
    [0,0,3],
    [1,1,0],
    [0,2,1],
    [1,0,2]]
)#训练集
res = enc.transform([[0,1,3]]).toarray() #将结果转化为数组
print(res)
