from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(X_train, y_train)

    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train, y_train)

    model_gb = GradientBoostingRegressor(random_state=42)
    model_gb.fit(X_train, y_train)

    return model_lr, model_dt, model_rf, model_gb

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def evaluate_all_models(models, X_test, y_test):
    model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
    for model, name in zip(models, model_names):
        mae, mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"{name} - MAE: {mae}, MSE: {mse}, R2: {r2}")