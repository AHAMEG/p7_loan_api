# Load librairies
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
from sklearn.neighbors import NearestNeighbors
import shap
# -------------------------------------------------------------------------------------------
#                                           loadings
# -------------------------------------------------------------------------------------------
# Model loading
# -------------
# path = os.path.join('Data', 'scoring_credit_model.pkl')
# with open(path, 'rb') as file:
#     bestmodel = joblib.load(file)
#
bestmodel = joblib.load('scoring_credit_model.pkl')

# # threshold loading
# # -------------
# path = os.path.join('threshold_model.pkl')
# with open(path, 'rb') as file:
#     threshold = joblib.load(file)

threshold = joblib.load('threshold_model.pkl')

# # dict_cleaned loading
# # -------------
# path = os.path.join('dict_cleaned.pkl')
# with open(path, 'rb') as file:
#     dict_cleaned = joblib.load(file)

# dict_cleaned = joblib.load('dict_cleaned.pkl')

# Data save :


# data loading
# -------------
X_test = joblib.load('X_test_sample_v2.csv')
X_train = joblib.load('X_train_sample_v2.csv')
y_test = joblib.load('y_test_sample_v2.csv')
y_train = joblib.load('y_train_sample_v2.csv')


x_train_sample = X_train.iloc[0: 100]
y_train_sample = y_train.iloc[0: 100]

# X_test = dict_cleaned['X_test']
# X_train = dict_cleaned['X_train']
# y_test = dict_cleaned['y_test']
# y_train = dict_cleaned['y_train']
# x_train_sample = X_train.iloc[0: 10000]
# y_train_sample = y_train.iloc[0: 10000]
# x_test_sample = X_test.iloc[0: 10000]
# y_test_sample = y_test.iloc[0: 10000]

X = pd.concat([X_test, X_train], axis=0)
y = pd.concat([y_test, y_train], axis=0)
# X = pd.concat([x_test_sample, y_test_sample], axis=0)
# y = pd.concat([y_test_sample, y_train_sample], axis=0)
# get the name of the columns
preproc_cols = X_train.columns

###############################################################
# instantiate Flask object
app = Flask(__name__)

@app.route("/")
def index():
    return "APP loaded, model and data loaded............"


# Customers id list  (
@app.route('/app/id/')
def ids_list():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    customers_id_list = pd.Series(list(X.index.sort_values()))  # X_test
    # Convert pd.Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': customers_id_list_json})


# test local : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=165690
@app.route('/app/data_cust/')  # ==> OK
def selected_cust_data():  # selected_id
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    x_cust = X.loc[selected_id_customer: selected_id_customer]  # X_test
    y_cust = y.loc[selected_id_customer: selected_id_customer] # y_test
    # Convert pd.Series to JSO
    data_x_json = json.loads(x_cust.to_json())
    y_cust_json = json.loads(y_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'y_cust': y_cust_json,
                    'data': data_x_json})


# find 20 nearest neighbors among the training set
def get_df_neigh(selected_id_customer):
    # fit nearest neighbors among the selection
    NN = NearestNeighbors(n_neighbors=20)
    NN.fit(X_train)  # X_train_NN
    X_cust = X.loc[selected_id_customer: selected_id_customer]  # X_test
    idx = NN.kneighbors(X=X_cust,
                        n_neighbors=20,
                        return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    # data and target of neighbors
    # ----------------------------
    x_neigh = X_train.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]

    return x_neigh, y_neigh

# Test local : http://127.0.0.1:5000/app/neigh_cust/?SK_ID_CURR=165690
@app.route('/app/neigh_cust/')  # ==> OK
def neigh_cust():
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    # Parse the http request to get arguments (selected_id), return the nearest neighbors
    data_neigh, y_neigh = get_df_neigh(selected_id_customer)
    # Convert to JSON
    data_neigh_json = json.loads(data_neigh.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'y_neigh':  y_neigh_json,
                    'data_neigh': data_neigh_json},  # 'x_cust': x_cust_json},
                   )


# find 10000 nearest neighbors among the training set
def get_df_thousand_neigh(selected_id_customer):
    # fit nearest neighbors among the selection
    thousand_nn = NearestNeighbors(n_neighbors=500)  # len(X_train)
    thousand_nn.fit(X_train)  # X_train_NN
    X_cust = X.loc[selected_id_customer: selected_id_customer]   # X_test
    idx = thousand_nn.kneighbors(X=X_cust,
                                 n_neighbors=500,  # len(X_train)
                                 return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    # data and target of neighbors
    # ----------------------------
    x_thousand_neigh = X_train.loc[nearest_cust_idx, :]
    y_thousand_neigh = y_train.loc[nearest_cust_idx]
    return x_thousand_neigh, y_thousand_neigh, X_cust


@app.route('/app/thousand_neigh/')  # ==> ok
# get shap values of the customer and 20 nearest neighbors
# Test local : http://127.0.0.1:5000/app/thousand_neigh/?SK_ID_CURR=165690
def thous_neigh():
    # Parse http request to get arguments (sk_id_cust)
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    x_thousand_neigh, y_thousand_neigh, x_customer = get_df_thousand_neigh(selected_id_customer)
    # Converting the pd.Series to JSON
    x_thousand_neigh_json = json.loads(x_thousand_neigh.to_json())
    y_thousand_neigh_json = json.loads(y_thousand_neigh.to_json())
    x_customer_json = json.loads(x_customer.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'X_thousand_neigh': x_thousand_neigh_json,
                    'x_custom': x_customer_json,
                    'y_thousand_neigh': y_thousand_neigh_json})


@app.route('/app/shap_val/')  # ==> ok
# get shap values of the customer and 20 nearest neighbors
# Test local : http://127.0.0.1:5000/app/shap_val/?SK_ID_CURR=165690
def shap_value():
    # Parse http request to get arguments (sk_id_cust)
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh, y_neigh = get_df_neigh(selected_id_customer)
    X_cust = X.loc[selected_id_customer: selected_id_customer]  # X_test
    # prepare the shap values of nearest neighbors + customer
    shap.initjs()
    # creating the TreeExplainer with our model as argument
    explainer = shap.TreeExplainer(bestmodel)  # X_train_2.sample(1000)
    # Expected values
    expected_vals = pd.Series(list(explainer.expected_value))
    # calculating the shap values of selected customer
    shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust)[1]))
    # calculating the shap values of neighbors
    shap_val_neigh_ = pd.Series(list(explainer.shap_values(X_neigh)[1]))  # shap_vals[1][X_neigh.index]
    # Converting the pd.Series to JSON
    X_neigh_json = json.loads(X_neigh.to_json())
    expected_vals_json = json.loads(expected_vals.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())  # json.loads(shap_val_neigh_.to_json())
    shap_vals_cust_json = json.loads(shap_vals_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'X_neigh_': X_neigh_json,
                    'shap_val_neigh': shap_val_neigh_json,  # double liste
                    'expected_vals': expected_vals_json,  # liste
                    'shap_val_cust': shap_vals_cust_json})  # double liste


# return all data of training set when requested
# Test local : http://127.0.0.1:5000/app/all_proc_data_tr/
@app.route('/app/all_proc_data_tr/')  # ==> OK
def all_proc_data_tr():
    # get all data from X_train, X_test and y_train data
    # and convert the data to JSON
    X_train_json = json.loads(x_train_sample.to_json())
    y_train_json = json.loads(y_train_sample.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'X_train': X_train_json,
                    'y_train': y_train_json})


# answer when asking for score and decision about one customer
# Test local : http://127.0.0.1:5000/app/scoring_cust/?SK_ID_CURR=165690

@app.route('/app/scoring_cust/')  # == > OK
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    customer_id = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X.loc[customer_id:customer_id] # X_test
    # X_cust = X_cust.drop(["SK_ID_CURR"], axis=1)
    # Compute the score of the customer (using the whole pipeline)
    score_cust = bestmodel.predict_proba(X_cust)[:, 1][0]
    # Return score
    return jsonify({'status': 'ok',
                    'SK_ID_CURR': customer_id,
                    'score': score_cust,
                    'thresh': threshold})


# return json object of feature importance (lgbm attribute)
# Test local : http://127.0.0.1:5000/app/feat

@app.route('/app/feat/')
def features():
    feat = X_test.columns
    f = pd.Series(feat)
    # Convert pd.Series to JSON
    feat_json = json.loads(f.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': feat_json})


# return json object of feature importance (lgbm attribute)
# Test local : http://127.0.0.1:5000/app/feat_imp

@app.route('/app/feat_imp/')
def send_feat_imp():
    feat_imp = pd.Series(bestmodel.feature_importances_,
                         index=X_test.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
                    'data': feat_imp_json})


# test
if __name__ == "__main__":
    app.run()
