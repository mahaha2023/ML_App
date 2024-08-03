import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import plotly.figure_factory as ff
import numpy as np

# Fonction pour charger les données
@st.cache_data
def load_data(file, sep, header, file_type):
    if file_type == 'csv':
        return pd.read_csv(file, sep=sep, header=header)
    elif file_type == 'excel':
        return pd.read_excel(file, header=header)
    elif file_type == 'text':
        return pd.read_csv(file, sep=sep, header=header)  # Assuming space or tab delimited
    else:
        st.error("Type de fichier non pris en charge.")
        return None

# Fonction pour encoder les colonnes en fonction des méthodes sélectionnées
@st.cache_data
def encode_categorical(data, columns_methods):
    encoded_data = data.copy()
    
    for col, methods in columns_methods.items():
        if 'OneHotEncoder' in methods:
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded_features = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))
            encoded_data = pd.concat([encoded_data, encoded_df], axis=1).drop(col, axis=1)
        if 'OrdinalEncoder' in methods:
            encoder = OrdinalEncoder()
            encoded_data[col] = encoder.fit_transform(data[[col]])
        if 'LabelEncoder' in methods:
            encoder = LabelEncoder()
            encoded_data[col] = encoder.fit_transform(data[col])
            
    return encoded_data

# Fonction pour traiter les valeurs manquantes
@st.cache_data
def handle_missing_values(data, columns_methods):
    imputed_data = data.copy()
    for col, method in columns_methods.items():
        if method == 'KNN Imputer':
            encoded_data = encode_categorical(data[[col]], {col: ['OrdinalEncoder']})
            imputer = KNNImputer(n_neighbors=5)
            imputed_data[col] = imputer.fit_transform(encoded_data)[..., 0]
        elif method.startswith('Valeur Spécifique'):
            value = float(method.split(':')[1].strip())
            imputed_data[col].fillna(value, inplace=True)
        elif method != 'Aucune':
            strategy_mapping = {
                'Moyenne': 'mean',
                'Médiane': 'median',
                'Mode': 'most_frequent'
            }
            imputer = SimpleImputer(strategy=strategy_mapping[method])
            imputed_data[[col]] = imputer.fit_transform(data[[col]])
    return imputed_data

# Fonction pour afficher les doublons
def show_duplicates(data):
    duplicates = data[data.duplicated()]
    return duplicates

# Fonction pour supprimer les doublons
def remove_duplicates(data):
    return data.drop_duplicates()

# Fonction pour afficher le pourcentage de données manquantes
def show_missing_percentage(data):
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    return missing_percentage

# Fonction pour supprimer des lignes ou colonnes
def remove_rows_or_columns(data, axis, indices):
    if axis == 'rows':
        return data.drop(indices, axis=0)
    elif axis == 'columns':
        return data.drop(indices, axis=1)
    else:
        st.error("Axe non pris en charge.")
        return data

# Fonction pour visualiser les valeurs manquantes
@st.cache_data
def plot_missing_values(data):
    fig = ff.create_annotated_heatmap(
        z=data.isnull().astype(int).values,
        x=list(data.columns),
        y=list(data.index),
        annotation_text=data.isnull().astype(int).values,
        colorscale='viridis'
    )
    return fig

# Fonction pour évaluer les modèles de classification
def evaluate_classification_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    return accuracy, precision, recall, f1, auc

# Fonction pour évaluer les modèles de régression
def evaluate_regression_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

# Interface Streamlit
st.title("Chargement, Traitement et Modélisation des Données")

# Widgets Streamlit pour les fichiers
st.write("Choisissez les fichiers pour les ensembles de données:")
train_file = st.file_uploader("Fichier d'entraînement (CSV, Excel ou Texte)", type=["csv", "xlsx", "txt"], key="train")
test_file = st.file_uploader("Fichier de test (CSV, Excel ou Texte)", type=["csv", "xlsx", "txt"], key="test")
eval_file = st.file_uploader("Fichier d'évaluation (CSV, Excel ou Texte)", type=["csv", "xlsx", "txt"], key="eval")

file_type_mapping = {
    "csv": "csv",
    "xlsx": "excel",
    "txt": "text"
}

# Fonction pour traiter les fichiers
def process_files(train_file, test_file, eval_file):
    data_train = None
    data_test = None
    data_eval = None

    if train_file:
        file_type = file_type_mapping.get(train_file.name.split('.')[-1])
        data_train = load_data(train_file, sep=",", header=0, file_type=file_type)
    
    if test_file:
        file_type = file_type_mapping.get(test_file.name.split('.')[-1])
        data_test = load_data(test_file, sep=",", header=0, file_type=file_type)
    
    if eval_file:
        file_type = file_type_mapping.get(eval_file.name.split('.')[-1])
        data_eval = load_data(eval_file, sep=",", header=0, file_type=file_type)
    
    return data_train, data_test, data_eval

# Charge les données lorsque les fichiers sont fournis
data_train, data_test, data_eval = process_files(train_file, test_file, eval_file)

if data_train is not None:
    st.write("Données d'entraînement chargées:")
    st.write(data_train.head())

if data_test is not None:
    st.write("Données de test chargées:")
    st.write(data_test.head())

if data_eval is not None:
    st.write("Données d'évaluation chargées:")
    st.write(data_eval.head())

if data_train is not None and data_test is not None:
    # Afficher un aperçu du pourcentage de données manquantes pour les données d'entraînement
    st.write("Pourcentage de valeurs manquantes par colonne dans les données d'entraînement:")
    missing_percentage_train = show_missing_percentage(data_train)
    st.write(missing_percentage_train)
    
    # Visualisation des valeurs manquantes pour les données d'entraînement
    st.write("Visualisation des valeurs manquantes pour les données d'entraînement:")
    fig_train = plot_missing_values(data_train)
    st.plotly_chart(fig_train)
    
    # Choix des méthodes d'encodage
    st.write("Choisissez les méthodes d'encodage pour chaque colonne:")
    columns_methods = {}
    for col in data_train.select_dtypes(include=['object']).columns:
        methods = st.multiselect(
            f"Colonnes à encoder - {col}",
            ['OneHotEncoder', 'OrdinalEncoder', 'LabelEncoder', 'Aucune'],
            default=['Aucune']
        )
        columns_methods[col] = methods
    
    if st.button("Appliquer les encodages"):
        data_encoded_train = encode_categorical(data_train, columns_methods)
        st.write("Données après encodage:")
        st.write(data_encoded_train.head())
        
        # Traitement des valeurs manquantes après encodage
        st.write("Choisissez les méthodes d'imputation pour chaque colonne:")
        imputation_methods = {}
        for col in data_encoded_train.columns:
            method = st.selectbox(
                f"Colonne {col}",
                ['Aucune', 'Moyenne', 'Médiane', 'Mode', 'KNN Imputer', 'Valeur Spécifique'],
                key=col
            )
            imputation_methods[col] = method
        
        if st.button("Appliquer les imputations"):
            data_imputed_train = handle_missing_values(data_encoded_train, imputation_methods)
            st.write("Données après imputation des valeurs manquantes:")
            st.write(data_imputed_train.head())
            
            # Affichage des doublons
            st.write("Affichage des doublons dans les données:")
            duplicates = show_duplicates(data_imputed_train)
            st.write(duplicates)
            
            if st.button("Supprimer les doublons"):
                data_imputed_train = remove_duplicates(data_imputed_train)
                st.write("Données après suppression des doublons:")
                st.write(data_imputed_train.head())
            
            # Suppression des lignes ou colonnes spécifiques
            st.write("Choisissez de supprimer des lignes ou des colonnes:")
            axis = st.selectbox("Axe", ["rows", "columns"])
            indices = st.text_input(f"Indices à supprimer ({axis})", "")
            
            if st.button("Supprimer"):
                indices_list = [int(idx) for idx in indices.split(',') if idx.strip().isdigit()]
                data_imputed_train = remove_rows_or_columns(data_imputed_train, axis, indices_list)
                st.write(f"Données après suppression des {axis}:")
                st.write(data_imputed_train.head())
                
            # Choix des modèles de classification
            st.write("Choisissez un modèle pour la classification:")
            classifier_model = st.selectbox("Modèle de classification", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"])
            
            if classifier_model == "Random Forest":
                model = RandomForestClassifier()
            elif classifier_model == "Logistic Regression":
                model = LogisticRegression()
            elif classifier_model == "Decision Tree":
                model = DecisionTreeClassifier()
            elif classifier_model == "SVM":
                model = SVC()
            
            # Entraînement et évaluation du modèle de classification
            if st.button("Entraîner et évaluer le modèle de classification"):
                X = data_imputed_train.drop(columns='target')  # Remplacez 'target' par le nom de la colonne cible
                y = data_imputed_train['target']  # Remplacez 'target' par le nom de la colonne cible
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy, precision, recall, f1, auc = evaluate_classification_model(y_test, y_pred)
                st.write("Évaluation du modèle de classification:")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")
                st.write(f"AUC: {auc:.2f}")

            # Choix des modèles de régression
            st.write("Choisissez un modèle pour la régression:")
            regressor_model = st.selectbox("Modèle de régression", ["Random Forest Regressor", "Linear Regression"])
            if regressor_model == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif regressor_model == "Linear Regression":
                model = LinearRegression()
            
            # Entraînement et évaluation du modèle de régression
            if st.button("Entraîner et évaluer le modèle de régression"):
                X = data_imputed_train.drop(columns='target')  # Remplacez 'target' par le nom de la colonne cible
                y = data_imputed_train['target']  # Remplacez 'target' par le nom de la colonne cible
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse, rmse, r2 = evaluate_regression_model(y_test, y_pred)
                st.write("Évaluation du modèle de régression:")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"R^2 Score: {r2:.2f}")
