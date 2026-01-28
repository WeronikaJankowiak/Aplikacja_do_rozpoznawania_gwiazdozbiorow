import joblib
import os
import pandas as pd
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

dataset_path = Path(__file__).resolve().parent / ".." / "b_image_processing" / "DataSet_1000.csv"
model_path = './c_stars_recognizing/model_test.pkl' # aby nadpisać któryś z modeli trzeba tu wstawić jego nazwę

def ReadDataset():
    """
    Wczytuje dataset i dzieli na zbiór treningowy i testowy.
    """
    dataset = pd.read_csv(dataset_path, encoding='ansi')
    
    # Wyświetlenie podstawowych informacji o datasecie
    print(f"Rozmiar datasetu: {dataset.shape}")
    print(f"Liczba unikalnych gwiazd (klas): {dataset['star_id'].nunique()}")
    print(f"Cechy: {dataset.drop(columns=['star_id']).columns.tolist()}")
    
    # Podział na cechy i etykiety
    X = dataset.drop(columns=['star_id'])
    y = dataset['star_id']
    
    # Sprawdzenie rozkładu klas
    class_counts = y.value_counts()
    print(f"\nStatystyki rozkładu klas:")
    print(f"  Min próbek na klasę: {class_counts.min()}")
    print(f"  Max próbek na klasę: {class_counts.max()}")
    print(f"  Średnia próbek na klasę: {class_counts.mean():.2f}")
    
    # Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Ważne bo zachowuje proporcje klas w train/test
    )
    
    return X_train, X_test, y_train, y_test

def CreateModel(model_type):
    """
    Tworzy i trenuje model według wybranego typu.
    
    Args:
        model_type: 'knn', 'rf', 'dt'
    """
    start_time = time.time()
    
    # Wczytanie danych
    X_train, X_test, y_train, y_test = ReadDataset()
    
    print(f"\n{'='*70}")
    print(f"Trenowanie modelu {model_type.upper()}")
    print('='*70)
    
    # Wybór modelu i odpowiednich danych [TECH] - parametr techniczny, [WYNIKI] - parametr wpływający na wyniki modelu
    if model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=1,  # [WYNIKI] liczba sąsiadów do klasyfikacji
            n_jobs=-1  # [TECH] użycie wszystkich rdzeni CPU
        )
        
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=25,  # [WYNIKI] liczba drzew w lesie
            max_depth=20,  # [WYNIKI] maksymalna głębokość drzewa
            min_samples_split=20,  # [WYNIKI] minimalna liczba próbek do podziału węzła
            min_samples_leaf=10,  # [WYNIKI] minimalna liczba próbek w liściu
            max_features='sqrt',  # [WYNIKI] liczba cech do rozważenia przy podziale
            max_samples=0.5,  # [WYNIKI] frakcja próbek do trenowania każdego drzewa
            n_jobs=1,  # [TECH] liczba rdzeni CPU
            random_state=42,  # [TECH] ziarno losowości
            verbose=1  # [TECH] wyświetlanie postępu
        )
        
    elif model_type == 'dt':
        model = DecisionTreeClassifier(
            max_depth=25,  # [WYNIKI] maksymalna głębokość drzewa
            min_samples_split=10,  # [WYNIKI] minimalna liczba próbek do podziału węzła
            min_samples_leaf=5,  # [WYNIKI] minimalna liczba próbek w liściu
            random_state=42  # [TECH] ziarno losowości
        )
        
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")
    
    # Trening modelu
    print(f"\nTrenowanie na {len(X_train)} próbkach...")
    train_start = time.time()
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - train_start
    print(f"Trenowanie zakończone w {train_time:.2f}s")
    
    # Predykcja
    print(f"\nPredykcja na {len(X_test)} próbkach...")
    predict_start = time.time()
    
    y_pred = model.predict(X_test)
    
    predict_time = time.time() - predict_start
    
    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Podsumowanie
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("WYNIKI")
    print('='*70)
    print(f"Dokładność:          {accuracy:.4f}")
    print(f"F1 (ważone):         {f1_weighted:.4f}")
    print(f"F1 (macro):          {f1_macro:.4f}")
    print(f"Czas trenowania:     {train_time:.2f}s")
    print(f"Czas predykcji:      {predict_time:.2f}s")
    print(f"Czas całkowity:      {total_time:.2f}s")
    print('='*70)
    
    # Zapisanie modelu
    model_file = model_path.replace('.pkl', f'_{model_type}.pkl')
    joblib.dump(model, model_file)
    print(f"\nModel zapisany do: {model_file}")
    
    return model

def LoadModel(path=model_path):
    """Wczytuje model"""
    if not os.path.isfile(path):
        return None
    
    model = joblib.load(path)
    return model


if __name__ == "__main__":
    model = CreateModel(model_type='knn')  # 'knn', 'rf', 'dt'
    