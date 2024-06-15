import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Judul dan deskripsi aplikasi
st.title("Analisis Asosiasi oleh Mas Nasrun dan Mas Ajik pakai K")
st.write("Aplikasi ini memungkinkan Anda untuk mengunggah file data (Excel atau CSV), mengatur nilai minimum confidence dan support, serta melakukan analisis asosiasi pada data tersebut.")

# Upload file
upload_file = st.file_uploader("Unggah file data (Excel atau CSV)", type=["csv", "xlsx"])

if upload_file is not None:
    # Membaca data dari file
    if upload_file.name.endswith('.csv'):
        data = pd.read_csv(upload_file, sep=',', encoding='utf-8')
    else:
        data = pd.read_excel(upload_file)

    # Input min_support dan min_confidence
    min_support = st.slider("Minimum Support", 0.0, 1.0, 0.2, 0.01)
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.01)

    # Konversi data menjadi matriks biner
    encoded_data = pd.get_dummies(data)
    encoded_data = encoded_data >= 1  # Mengonversi data menjadi format biner

    # Menghitung frekuensi itemset
    frequent_itemsets = apriori(encoded_data.astype('bool'), min_support=min_support, use_colnames=True)
    st.write("Frequent Itemsets:")
    st.write(frequent_itemsets)

    # Menghitung aturan asosiasi
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    st.write("Aturan Asosiasi:")
    st.write(rules)

else:
    st.write("Silakan unggah file data (Excel atau CSV) terlebih dahulu.")