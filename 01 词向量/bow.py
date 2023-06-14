from sklearn.feature_extraction.text import CountVectorizer
texts=['橘子 香蕉 苹果 葡萄','葡萄 苹果 苹果','葡萄','橘子 苹果'] #语料库
cv = CountVectorizer() #词袋模型对象
cv_fit = cv.fit_transform(texts) #完成文本到向量的表示
print(cv.vocabulary_) #词汇表
print(cv_fit.toarray()) #文本向量表示的数组格式
