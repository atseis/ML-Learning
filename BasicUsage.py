import pandas as pd

#%%----------------------------------------------------------------
# 从文件中读取数据
filename = './melb_data.csv'
data = pd.read_csv(filename)

#%%----------------------------
# 查看数据信息
data.describe()
#%%----------------------------------------------------------------
# 从整体数据中提取训练集、验证集
features =['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X=data[features]
Y=data.Price
from sklearn.model_selection import train_test_split
train_X, val_X, train_Y, val_Y=train_test_split(X,Y, random_state=0)

#%%----------------------------------------------------------------
# 训练模型
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_Y)

#%%----------------------------
# 验证模型
from sklearn.metrics import mean_absolute_error
predictions=model.predict(val_X)
print(mean_absolute_error(val_Y, predictions))



#%%----------------------------
# Use Random Forest
from sklearn.ensemble import RandomForestRegressor
forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_Y)
forest_predictions=forest_model.predict(val_X)
print(mean_absolute_error(val_Y, forest_predictions))