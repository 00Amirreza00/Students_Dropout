import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE , ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

st.image('picture1.jpg')
st.title('Students\' Dropout and Academic Success')
st.write('The dataset created from a higher education institution related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies.')
st.write('The dataset includes information known at the time of student enrollment (academic path, demographics, and social-economic factors) and the students\' academic performance at the end of the first and second semesters.')
st.write('The data is used to build classification models to predict students\' dropout and academic success.')
st.write('The problem is formulated as a three category classification task, in which there is a strong imbalance towards one of the classes.')

st.write('The dataset is available on UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success')


df = pd.read_csv('data.csv', delimiter=';')
df.rename(columns={'Nacionality': 'Nationality','Daytime/evening attendance\t': 'Daytime/evening attendance'}, inplace=True) 


categorical_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification', 
                    "Nationality", "Mother's qualification", "Father's qualification", 
                    "Mother's occupation", "Father's occupation"]

continuous_cols = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP']

# Convert categorical features
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Convert binary features
binary_cols = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 
               'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
for col in binary_cols:
    df[col] = df[col].astype(bool)

# Ordinal column
ordinal_cols = ['Application order']

#target column
target_col = 'Target'

# Calculate imbalance ratio
class_counts = df['Target'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()

######
st.subheader('Data Exploration')
with st.expander("See explanation"):
    st.write('This section provides an overview of the dataset, including its shape, class imbalance, and basic statistics.')
    st.write('The dataset contains {} students and {} features.'.format(df.shape[0], df.shape[1]))
    st.write('The dataset contains {} categorical columns, {} binary columns, and {} numerical columns.'.format(len(categorical_cols), len(binary_cols), len(continuous_cols) + len(ordinal_cols)))
    st.write('The dataset is relatively clean, with no missing values in the features.')
    st.write('The target variable is "Target", which indicates the students\' dropout and academic success status.')
    plt.figure(figsize=(4, 3))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Class Distribution in Target Variable')  
    st.pyplot(plt)
    st.write('The class distribution in the target variable is highly imbalanced, with the majority class being "Graduate".')
    st.write('The imbalance ratio is {:.2f}, indicating a significant imbalance between the classes.'.format(imbalance_ratio))

######
st.subheader('Data Modelling and Evaluation')
with st.expander("Metrics"):
    st.write('The conventional metrics like accuracy, precision, recall, and F1 score are not suitable for imbalanced datasets, as they can be misleading.')
    st.write('The models are evaluated using the following metrics:')
    st.write('- Macro-averaged F1 score: This metric is used to evaluate the models\' performance on imbalanced datasets, as it takes into account the precision and recall for each class and averages them equally.')
    st.write('It averages F1 (harmonic mean of precision & recall) across all classes, unweighted')
    st.write('Directly reflects per-class performance, not biased by class size. And, it penalizes poor performance on any class equally, regardless of class size')
    st.write('- Confusion matrix: This matrix is used to visualize the performance of the models, showing the true positive, true negative, false positive, and false negative counts for each class.')


with st.expander("Data Modelling"):
    st.write('The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.')
    st.write('For the basic modeling, I use a KNN classifier with k=3.')
    st.write('Then I apply different sampling techniques to address the class imbalance problem, including SMOTE, Random Over Sampling (ROS), Random Under Sampling (RUS), and ADASYN.')
    st.write('Then I try to study an ensemble learning model like Random Forest Classifier (RFC) combined with the sampling techniques.')
    st.write('I also considered a RFC with weighted sampling. It Trains a standard random forest.\n During training, each sample is weighted according to its class.\n The algorithm tries to minimize errors on minority classes by giving them more importance in the split criteria.')
    st.write('The sampling techniques used in this project are:')
    tab1, tab2, tab3, tab4 = st.tabs(["SMOTE", "ROS", "RUS", "ADASYN"])
    with tab1:
        st.write('SMOTE (Synthetic Minority Over-sampling Technique) is a popular technique for addressing class imbalance by generating synthetic samples for the minority class.')
        st.write('This technique generates synthetic samples for the minority classes by interpolating between existing samples.')
        st.image('smote.jpg', caption='SMOTE technique illustration', use_column_width=True)
    with tab2:
        st.write('Random Over Sampling (ROS) is a simple technique that randomly duplicates samples from the minority classes to balance the class distribution.')    
        st.write('This technique randomly duplicates samples from the minority classes to balance the class distribution.')
        st.image('ros.jpg', caption='Random Over Sampling (ROS) technique illustration', use_column_width=True)
    with tab3:
        st.write('Random Under Sampling (RUS) is a technique that randomly removes samples from the majority class to balance the class distribution.')
        st.write('This technique randomly removes samples from the majority class to balance the class distribution.')
        st.image('rus.jpg', caption='Random Under Sampling (RUS) technique illustration', use_column_width=True)
    with tab4:
        st.write('ADASYN (Adaptive Synthetic Sampling) is an advanced technique that generates synthetic samples for the minority classes, focusing more on the difficult-to-learn samples.')
        st.write('This technique generates synthetic samples for the minority classes, focusing more on the difficult-to-learn samples.')
        st.image('adasyn.jpg', caption='ADASYN technique illustration', use_column_width=True)
   
     
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train0, X_test, y_train0, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)    

####################
    Model_names = ['RFC', 'RFC_with_weighted_sampling', 'SMOTE', 'ROS', 'RUS', 'ADASYN']
    all_results = []

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough')

    # Fit and transform training and test data once
    X_train0_transformed = preprocessor.fit_transform(X_train0)
    X_test_transformed = preprocessor.transform(X_test)

    pipeline = RandomForestClassifier(random_state=42)

    for model in Model_names:
        if model == 'RFC':
            X_train = X_train0_transformed
            y_train = y_train0
        elif model == 'SMOTE':
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train, y_train = smote.fit_resample(X_train0_transformed, y_train0)
        elif model == 'ROS':
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train0_transformed, y_train0)
        elif model == 'RUS':
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train0_transformed, y_train0)
        elif model == 'ADASYN':
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train0_transformed, y_train0)
            
        if  model == 'RFC_with_weighted_sampling':
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train0)
            pipeline.fit(X_train0_transformed, y_train0, sample_weight=sample_weights)
            y_pred = pipeline.predict(X_test_transformed) 
        else:  
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test_transformed)
            
        all_results.append({
                "type": model,
                "macro_averaged_F1_score": f1_score(y_test, y_pred, average='macro'),
                "confusion_matrix": confusion_matrix(y_test, y_pred, normalize='true'),
            })

    n_models = len(all_results)
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))

    df_results = pd.DataFrame(all_results).sort_values("macro_averaged_F1_score", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten() 

    for idx, result in df_results.iterrows():
        disp = ConfusionMatrixDisplay(confusion_matrix=result["confusion_matrix"], display_labels=le.classes_)
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(f'{result["type"]}\nMacro-averaged F1 score: {result["macro_averaged_F1_score"]:.3f}')

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()


#########################################
    all_results_knn=[]

    ###
    preprocessor_knn = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough')

    pipeline_knn = Pipeline([
        ('preprocess', preprocessor_knn),
        ('clf', KNeighborsClassifier(n_neighbors=3))])

    pipeline_knn.fit(X_train0, y_train0)
    y_pred = pipeline_knn.predict(X_test)

    all_results_knn.append({
                "type": "knn_without_normalization",
                "macro_averaged_F1_score": f1_score(y_test, y_pred, average='macro'),
                "confusion_matrix": confusion_matrix(y_test, y_pred, normalize='true'),
            })
    ###

    preprocessor_knn = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('cont', StandardScaler(), continuous_cols + ordinal_cols)],
        remainder='passthrough')

    pipeline_knn = Pipeline([
        ('preprocess', preprocessor_knn),
        ('clf', KNeighborsClassifier(n_neighbors=3))])

    Model_names_knn = ['knn', 'SMOTE', 'ROS', 'RUS', 'ADASYN']
    for model in Model_names_knn:
        if model == 'knn':
            X_train = X_train0
            y_train = y_train0
        elif model == 'SMOTE':
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train, y_train = smote.fit_resample(X_train0, y_train0)
        elif model == 'ROS':
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train0, y_train0)
        elif model == 'RUS':
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train0, y_train0)
        elif model == 'ADASYN':
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train0, y_train0)

        pipeline_knn.fit(X_train, y_train)
        y_pred = pipeline_knn.predict(X_test)
            
        all_results_knn.append({
                "type": model,
                "macro_averaged_F1_score": f1_score(y_test, y_pred, average='macro'),
                "confusion_matrix": confusion_matrix(y_test, y_pred, normalize='true'),
            })

    n_models = len(all_results_knn)
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))

    df_results_knn = pd.DataFrame(all_results_knn).sort_values("macro_averaged_F1_score", ascending=False).reset_index(drop=True)

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes1 = axes1.flatten() 

    for idx, result in df_results_knn.iterrows():
        disp = ConfusionMatrixDisplay(confusion_matrix=result["confusion_matrix"], display_labels=le.classes_)
        disp.plot(ax=axes1[idx], colorbar=False, cmap='Blues')
        axes1[idx].set_title(f'{result["type"]}\nMacro-averaged F1 score: {result["macro_averaged_F1_score"]:.3f}')

    for idx in range(n_models, len(axes1)):
        axes1[idx].axis('off')
    plt.tight_layout()

with st.expander("Results"):
    tab11,tab22, tab33 = st.tabs(["KNN", "RFC", "Results table"])

    with tab22:
        st.write('Here are the confusion matrices for the Random Forest Classifier (RFC) with different sampling techniques applied to the training data.')
        st.write('One of the models is the RFC with weighted sampling, which uses the sample weights to balance the class distribution. It is used to address the class imbalance problem by assigning weights to the samples based on their class distribution.')
        st.write('The models are evaluated using macro-averaged F1 score and confusion matrices.')
        st.write('The models are ordered by macro-averaged F1 score, with the best performing model at the first.')
        st.pyplot(fig)

    with tab11:
        st.write('Here are the confusion matrices for the KNN classifier with different sampling techniques applied to the training data.')
        st.write('One of the models is the KNN classifier with normalization, which uses the StandardScaler to normalize the continuous features. It is used to improve the classification problem by normalizing the continuous features.')
        st.write('The models are evaluated using macro-averaged F1 score and confusion matrices.')
        st.write('The models are ordered by macro-averaged F1 score, with the best performing model at the first.')
        st.pyplot(fig1)

    with tab33:
        st.write('Here is the table with the results of the models.')
        st.write('The table shows the macro-averaged F1 score for each model.')
        st.write('The models are ordered by macro-averaged F1 score, with the best performing model at the first.')
        st.caption('KNN Classifier Results')
        st.table(df_results_knn.drop(columns=['confusion_matrix']))
        st.caption('Random Forest Classifier Results')
        st.table(df_results.drop(columns=['confusion_matrix']))
        
st.subheader('Conclusion')
with st.expander("See explanation"):
    st.write('The Random Forest Classifier (RFC) with Random Oversampling has the highest macro-averaged F1 score of {:.3f}.'.format(df_results.iloc[0]['macro_averaged_F1_score']))
    st.write('The KNN classifier with SMOTE achieved the highest macro-averaged F1 score of {:.3f} among the other techniques with KNN.'.format(df_results_knn.iloc[0]['macro_averaged_F1_score']))
    st.write('The results show that the RFC with Random Oversampling performs better for this dataset.')
    st.write('The confusion matrices show that the models are able to correctly classify the majority of the instances, but there are still some misclassifications, especially for the minority classes.')
    st.write('Note that the chosen model is depend on the problem and the dataset, and it is important to evaluate the models using different metrics based on the problem at hand.')
