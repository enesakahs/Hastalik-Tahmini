from flask import Flask, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import sklearn
from flask import Flask, request, jsonify
from collections import Counter
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import LabelEncoder
import webbrowser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


app = Flask(__name__, template_folder='templates', static_folder='templates/static')
CORS(app)

app.config['DEBUG'] = False





print("System is starting")
def run_the_system():

    # Verileri X_train'deki sütun sayısıyla eşleşecek şekilde yeniden şekillendirdim
    X_train = ['Drying and tingling lips', 'Slurred speech', 'Swollen legs', 'Blurred and distorted vision', 'Abdominal pain',
     'Dizziness', 'Lethargy', 'Swollen blood vessels', 'Hip joint pain', 'Swelling joints', 'Foul smell of urine',
     'Stiff neck', 'Swelling of stomach', 'Obesity', 'Nodal skin eruptions', 'Receiving blood transfusion',
     'Passage of gases', 'Muscle wasting', 'Brittle nails', 'Ulcers on tongue', 'Watering from eyes', 'Cramps',
     'Yellowish skin', 'Malaise', 'Lack of concentration', 'Scurring', 'Spotting urination', 'Nausea', 'Fluid overload',
     'Pain during bowel movements', 'Irritability', 'Coma', 'Patches in throat', 'Blister', 'Irregular sugar level',
     'Itching', 'Fatigue', 'Constipation', 'Palpitations', 'Toxic look (typhos)']
    class_labels = [
        'Fungal infection', 'Hepatitis C', 'Hepatitis E', 'Alcoholic hepatitis',
        'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
        'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism',
        'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal Positional Vertigo',
        'Acne', 'Urinary tract infection', 'Psoriasis', 'Hepatitis D', 'Hepatitis B',
        'Allergy', 'hepatitis A', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
        'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma',
        'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
        'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Impetigo'
    ]
    diseases_list = ['Drying and tingling lips', 'Slurred speech', 'Swollen legs', 'Blurred and distorted vision',
                     'Abdominal pain', 'Dizziness', 'Lethargy', 'Swollen blood vessels', 'Hip joint pain',
                     'Swelling joints', 'Foul smell of urine', 'Stiff neck', 'Swelling of stomach', 'Obesity',
                     'Nodal skin eruptions', 'Receiving blood transfusion', 'Passage of gases', 'Muscle wasting',
                     'Brittle nails', 'Ulcers on tongue', 'Watering from eyes', 'Cramps', 'Yellowish skin', 'Malaise',
                     'Lack of concentration', 'Scurring', 'Spotting urination', 'Nausea', 'Fluid overload',
                     'Pain during bowel movements', 'Irritability', 'Coma', 'Patches in throat', 'Blister',
                     'Irregular sugar level', 'Itching', 'Fatigue', 'Constipation', 'Palpitations', 'Toxic look (typhos)']


    with open('C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/selected_disease.txt', 'r') as file:
        text = [disease.strip() for disease in file.read().split(',')]

    selected_diseases_list = [disease.strip() for disease in text if disease.strip()]


    # Her bir hastalığı anahtar ve ikili değerleri (0 veya 1) değer olarak içeren bir sözlük oluştur
    data = {disease: 1 if disease in selected_diseases_list else 0 for disease in diseases_list}

    # Tek satırlı bir sözlük listesinden bir DataFrame oluşturma
    df = pd.DataFrame([data])
    df = df.iloc[:,:40]

    regular = pd.read_csv("C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/Original_Dataset.csv") # Dataseti oku

    # Her bir değeri sütunlara dönüştürme
    columns_to_check = []
    for col in regular.columns:
        if col != 'Disease':
            columns_to_check.append(col)
    symptoms = regular.iloc[:, 1:].values.flatten()
    symptoms = list(set(symptoms))

    for symptom in symptoms:
        regular[symptom] = regular.iloc[:, 1:].apply(lambda row: int(symptom in row.values), axis=1)

    regular_1 = regular.drop(columns=columns_to_check)
    regular_1 = regular_1.loc[:, regular_1.columns.notna()]

    regular_1.columns = regular_1.columns.str.strip()

    # Gereksiz sütunları kaldırma
    regular.drop( ["Symptom_1","Symptom_2","Symptom_3","Symptom_4","Symptom_5","Symptom_6","Symptom_7","Symptom_8","Symptom_9","Symptom_10","Symptom_11","Symptom_12","Symptom_13","Symptom_14","Symptom_15","Symptom_16","Symptom_17"], axis = 1, inplace = True)



    lab = LabelEncoder()
    
    regular_1['Disease'] = lab.fit_transform(regular_1['Disease'])


    from sklearn.model_selection import train_test_split
    X = regular_1.drop("Disease", axis = 1)
    y = regular_1["Disease"]
    X = X = X.iloc[:, :40]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.2)



    X_train = np.array(X_train)  # X_train bir NumPy dizisi değilse, bir NumPy dizisine dönüştür
    y_train = np.array(y_train)  # NumPy dizisi değilse y_train'i bir NumPy dizisine dönüştür
    X_test = np.array(X_test)    # X_test, NumPy dizisi değilse NumPy dizisine dönüştür

    # Model (KNN)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # y_pred için Tahmin Oluşturma
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Verilen df için tahminde bulunma
    df=df.to_numpy()
    df = df.reshape(1, -1)[:, :40]  # Yalnızca ilk 40 sütunu kullan
    print("Shape of df:", df.shape)
    print("Shape of y_test:", y_test.shape)
    y_preds = knn.predict(df)

    print(y_preds)
    
    if y_preds == 0:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Fungual_infection/fungal_infection.html"
        return webbrowser.open(file_path)
    elif y_preds == 1:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hepatitis_c/Hepatitis_c.html"
        return webbrowser.open(file_path)
    elif y_preds ==2:
        file_path= r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hepatitis_e/Hepatitis_e.html"
        return webbrowser.open(file_path)
    elif y_preds == 3:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Alcoholic_hepatitis/Alcholoic_hepatitis.html"
        return webbrowser.open(file_path)
    elif y_preds == 4:
        file_path =  r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Turberculosis/Turberculosis.html"
        return webbrowser.open(file_path)
    elif y_preds == 5:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Common_cold/Common_cold.html"
        return webbrowser.open(file_path)
    elif y_preds == 6:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Pneumonia/Pneumonia.html"
        return webbrowser.open(file_path)
    elif y_preds == 7:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Dimorphic_hemmorhoids/Dimorphic_hemmorhoids.html"
        return webbrowser.open(file_path)
    elif y_preds == 8:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Heart_attack/Heart_attack.html"
        return webbrowser.open(file_path)
    elif y_preds == 9:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Varicose_veins/Varicose_veins.html"
        return webbrowser.open(file_path)
    elif y_preds == 10:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hypothyroidism/Hypothyroidism.html"
        return webbrowser.open(file_path)
    elif y_preds == 11:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hypothyroidism/Hypothyroidism.html"
        return webbrowser.open(file_path)
    elif y_preds == 12:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hypoglycemia/Hypoglycemia.html"
        return webbrowser.open(file_path)
    elif y_preds == 13:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Osteoarthristis/Osteoarthristis.html"
        return webbrowser.open(file_path)
    elif y_preds == 14:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Arthritis/Arthritis.html"
        return webbrowser.open(file_path)
    elif y_preds == 15:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Paroymsal_positional_vertigo/Paroymsal_positional_vertigo.html"
        return webbrowser.open(file_path)
    elif y_preds == 16:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Acne/Acne.html"
        return webbrowser.open(file_path)
    elif y_preds == 17:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Urinary_tract_infection/Urinary_tract_infection.html"
        return webbrowser.open(file_path)
    elif y_preds == 18:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Psoriasis/Psoriasis.html"
        return webbrowser.open(file_path)
    elif y_preds == 19:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hepatitis_d/Hepatitis_d.html"
        return webbrowser.open(file_path)
    elif y_preds == 20:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hepatitis_b/Hepatitis_b.html"
        return webbrowser.open(file_path)
    elif y_preds == 21:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Allergy/Allergy.html"
        return webbrowser.open(file_path)
    elif y_preds == 22:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hepatitis_a/Hepatitis_a.html"
        return webbrowser.open(file_path)
    elif y_preds == 23:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Gerd/Gerd.html"
        return webbrowser.open(file_path)
    elif y_preds == 24:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Chronic_cholestasis/Chronic_cholestasis.html"
        return webbrowser.open(file_path)
    elif y_preds == 25:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Drug_reaction/Drug_reaction.html"
        return webbrowser.open(file_path)
    elif y_preds == 26:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Peptic_ulcer_diseae/Peptic_ulcer_diseae.html"
        return webbrowser.open(file_path)
    elif y_preds == 27:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Aids/Aids.html"
        return webbrowser.open(file_path)
    elif y_preds == 28:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Diabetes/Diabetes.html"
        return webbrowser.open(file_path)
    elif y_preds == 29:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Gastroenteritis/Gastroenteritis.html"
        return webbrowser.open(file_path)
    elif y_preds == 30:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Bronchial_asthma/Bronchial_asthma.html"
        return webbrowser.open(file_path)
    elif y_preds == 31:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Hypertension/Hypertension.html"
        return webbrowser.open(file_path)
    elif y_preds == 32:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Migraine/Migraine.html"
        return webbrowser.open(file_path)
    elif y_preds == 33:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Cervical_spondylosis/Cervical_spondylosis.html"
        return webbrowser.open(file_path)
    elif y_preds == 34:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Paralysis/Paralysis.html"
        return webbrowser.open(file_path)
    elif y_preds == 35:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Jaundice/Jaundice.html"
        return webbrowser.open(file_path)
    elif y_preds == 36:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Malaria/Malaria.html"
        return webbrowser.open(file_path)
    elif y_preds == 37:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Chicken_pox/Chicken_pox.html"
        return webbrowser.open(file_path)
    elif y_preds == 38:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Dengue/Dengue.html"
        return webbrowser.open(file_path)
    elif y_preds == 39:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Typhoid/Typhoid.html"
        return webbrowser.open(file_path)
    elif y_preds == 40:
        file_path = r"C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/content/Impetigo/Impetigo.html"
        return webbrowser.open(file_path)


print("predict Class Initialized")

# Sınıf tahmini yapılacak API endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()  # Seçilen hastalıkları doğru şekilde aldığından emin ol
        selected_diseases = data.get('diseases', [])
        print('Received data:', selected_diseases)

        # Veriyi dosyaya yazma
        with open('C:/Users/enes_/OneDrive/Desktop/Python/Hastalık-Belirti-Tahmin/Hastalık/selected_disease.txt', 'w') as file:
            for disease in selected_diseases:
                file.write(disease + ',')

        # Tahmin yapmak için sistemi çalıştır
        predicted_label = run_the_system()

        # Tahmin sonucunu JSON yanıtı olarak döndür
        if predicted_label:
            return jsonify({'prediction': predicted_label})
        else:
            return jsonify({'prediction': 'Prediction failed'})

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug = True)