from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================================================
# LOAD PREPROCESSING FILES
# =========================================================

scaler = joblib.load(
    "scaler.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

# =========================================================
# LOAD ENCODER MODEL
# =========================================================

encoder = load_model(
    "encoder_model.keras"
)

# =========================================================
# LOAD HYBRID XGBOOST MODEL
# =========================================================

hybrid_xgb_model = XGBClassifier()

hybrid_xgb_model.load_model(
    "xgb_model.json"
)

# =========================================================
# LOAD STANDALONE XGBOOST MODEL
# =========================================================

standalone_xgb_model = XGBClassifier()

standalone_xgb_model.load_model(
    "xgb_model_standalone.pkl"
)

# =========================================================
# LOAD CATBOOST MODEL
# =========================================================

catboost_model = CatBoostClassifier()

catboost_model.load_model(
    "catboost_model_standalone.pkl"
)

# =========================================================
# LOAD ADABOOST MODEL
# =========================================================

adaboost_model = joblib.load(
    "adaboost_model_standalone.pkl"
)

# =========================================================
# LOAD RANDOM FOREST MODEL
# =========================================================

random_forest_model = joblib.load(
    "random_forest_smote_model.joblib"
)

# =========================================================
# LOAD CNN MODEL
# =========================================================

cnn_model = load_model(
    "cnn_ids_model.keras"
)

# =========================================================
# LOAD KNN MODEL
# =========================================================

knn_model = joblib.load(
    "knn_model.joblib"
)

# =========================================================
# LOAD SVM MODEL
# =========================================================

svm_model = joblib.load(
    "svm_model.joblib"
)

# =========================================================
# LOAD NAIVE BAYES MODEL
# =========================================================

naive_bayes_model = joblib.load(
    "naive_bayes_model.joblib"
)

print("ALL MODELS LOADED SUCCESSFULLY!")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# INPUT SCHEMA
# =========================================================

class NetworkData(BaseModel):

    duration: float

    protocol_type: str
    service: str
    flag: str

    src_bytes: float
    dst_bytes: float
    land: float
    wrong_fragment: float
    urgent: float
    hot: float
    num_failed_logins: float
    logged_in: float
    num_compromised: float
    root_shell: float
    su_attempted: float
    num_root: float
    num_file_creations: float
    num_shells: float
    num_access_files: float
    num_outbound_cmds: float
    is_host_login: float
    is_guest_login: float
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {

        "message": "IDS Multi-Model API Running"

    }

# =========================================================
# TREE MODEL PREPROCESSING
# =========================================================

def preprocess_tree(data):

    input_df = pd.DataFrame(
        [data.dict()]
    )

    # One Hot Encoding
    input_df = pd.get_dummies(
        input_df
    )

    # Match Training Columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df

# =========================================================
# HYBRID MODEL PREPROCESSING
# =========================================================

def preprocess_hybrid(data):

    input_df = preprocess_tree(
        data
    )

    scaled_input = scaler.transform(
        input_df
    )

    scaled_input = scaled_input.astype(
        np.float32
    )

    return scaled_input

# =========================================================
# GENERIC PREDICTION FUNCTION
# =========================================================

def make_prediction(model, features):

    prediction = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    predicted_label = label_encoder.inverse_transform(
        prediction.astype(int)
    )

    confidence = float(
        np.max(probabilities)
    )

    class_probabilities = {

        label_encoder.inverse_transform([i])[0]:
        round(float(prob), 4)

        for i, prob in enumerate(probabilities[0])

    }

    return {

        "prediction": predicted_label[0],

        "confidence": round(
            confidence,
            4
        ),

        "class_probabilities":
        class_probabilities

    }

# =========================================================
# KERAS PREDICTION FUNCTION
# =========================================================

def make_keras_prediction(model, features):

    probabilities = model.predict(
        features,
        verbose=0
    )

    predicted_class_idx = int(
        np.argmax(probabilities[0])
    )

    predicted_label = label_encoder.inverse_transform(
        [predicted_class_idx]
    )

    confidence = float(
        np.max(probabilities[0])
    )

    class_probabilities = {

        label_encoder.inverse_transform([i])[0]:
        round(float(prob), 4)

        for i, prob in enumerate(probabilities[0])

    }

    return {

        "prediction": predicted_label[0],

        "confidence": round(
            confidence,
            4
        ),

        "class_probabilities":
        class_probabilities

    }

# =========================================================
# SVM PREDICTION FUNCTION (USING SOFTMAX OVER DECISION FUNCTION)
# =========================================================

def make_svm_prediction(model, features):

    prediction = model.predict(
        features
    )

    decision_vals = model.decision_function(
        features
    )

    predicted_label = label_encoder.inverse_transform(
        prediction.astype(int)
    )

    # Apply Softmax to map decision values to pseudo-probabilities
    exp_vals = np.exp(decision_vals[0] - np.max(decision_vals[0]))
    probabilities = exp_vals / np.sum(exp_vals)

    confidence = float(
        np.max(probabilities)
    )

    class_probabilities = {

        label_encoder.inverse_transform([i])[0]:
        round(float(prob), 4)

        for i, prob in enumerate(probabilities)

    }

    return {

        "prediction": predicted_label[0],

        "confidence": round(
            confidence,
            4
        ),

        "class_probabilities":
        class_probabilities

    }

# =========================================================
# HYBRID AE + XGBOOST
# =========================================================

@app.post("/predict/hybrid")
def predict_hybrid(data: NetworkData):

    try:

        scaled_input = preprocess_hybrid(
            data
        )

        # Generate Encoded Features
        encoded_features = encoder.predict(
            scaled_input,
            verbose=0
        )

        encoded_features = encoded_features.astype(
            np.float32
        )

        # Feature Fusion
        fused_features = np.concatenate(

            [
                scaled_input,
                encoded_features
            ],

            axis=1
        )

        return make_prediction(

            hybrid_xgb_model,

            fused_features

        )

    except Exception as e:

        return {

            "error": str(e)

        }

# =========================================================
# STANDALONE XGBOOST
# =========================================================

@app.post("/predict/xgboost")
def predict_xgboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            standalone_xgb_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/catboost")
def predict_catboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            catboost_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/adaboost")
def predict_adaboost(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            adaboost_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/randomforest")
def predict_randomforest(data: NetworkData):

    try:

        tree_input = preprocess_tree(
            data
        )

        return make_prediction(

            random_forest_model,

            tree_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/cnn")
def predict_cnn(data: NetworkData):

    try:

        # Preprocess tree features
        tree_input = preprocess_tree(
            data
        )

        # Scale features
        scaled_input = scaler.transform(
            tree_input
        )

        # Reshape to (1, num_features, 1) for CNN
        cnn_input = scaled_input.reshape(
            scaled_input.shape[0],
            scaled_input.shape[1],
            1
        )

        cnn_input = cnn_input.astype(
            np.float32
        )

        return make_keras_prediction(

            cnn_model,

            cnn_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/knn")
def predict_knn(data: NetworkData):

    try:

        scaled_input = preprocess_hybrid(
            data
        )

        return make_prediction(

            knn_model,

            scaled_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/svm")
def predict_svm(data: NetworkData):

    try:

        scaled_input = preprocess_hybrid(
            data
        )

        return make_svm_prediction(

            svm_model,

            scaled_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }

@app.post("/predict/naivebayes")
def predict_naivebayes(data: NetworkData):

    try:

        scaled_input = preprocess_hybrid(
            data
        )

        return make_prediction(

            naive_bayes_model,

            scaled_input

        )

    except Exception as e:

        return {

            "error": str(e)

        }