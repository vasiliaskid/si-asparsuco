import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Definisi fungsi untuk halaman Upload Data
def upload_data():
    st.title("Upload Data")
    st.write("Unggah file data (Excel atau CSV) untuk melakukan analisis asosiasi.")

    # Upload file
    upload_file = st.file_uploader("Unggah file data (Excel atau CSV)", type=["csv", "xlsx"])

    if upload_file is not None:
        if upload_file.name.endswith('.csv'):
            data = pd.read_csv(upload_file, sep=',', encoding='utf-8')
        else:
            data = pd.read_excel(upload_file)

        st.write("Data yang diunggah:")
        st.dataframe(data)  # Menampilkan seluruh data yang diunggah dalam bentuk tabel

        # Simpan data ke session_state
        st.session_state['data'] = data

    elif 'data' in st.session_state:
        # Jika tidak ada file yang diunggah, tampilkan data dari session_state
        data = st.session_state['data']
        st.write("Data yang diunggah:")
        st.dataframe(data)

    else:
        st.write("Silakan unggah file data (Excel atau CSV) terlebih dahulu.")

# Definisi fungsi untuk halaman Analisis Asosiasi
def analisis_asosiasi():
    st.title("Analisis Asosiasi")

    # Cek apakah data sudah diunggah
    if 'data' not in st.session_state:
        st.write("Silakan unggah file data terlebih dahulu di halaman 'Upload Data'.")
    else:
        data = st.session_state['data']

        # Input min_support dan min_confidence
        min_support = st.slider("Minimum Support", 0.0, 1.0, 0.2, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.01)

        # Konversi data menjadi matriks biner
        encoded_data = pd.get_dummies(data)
        encoded_data = encoded_data >= 1  # Mengonversi data menjadi format biner

        # Menghitung frekuensi itemset
        frequent_itemsets = apriori(encoded_data.astype('bool'), min_support=min_support, use_colnames=True)
        st.write("Frequent Itemsets:")
        st.dataframe(frequent_itemsets)  # Menampilkan frequent itemsets dalam bentuk tabel

        # Menghitung aturan asosiasi
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        with st.expander("Lihat Aturan Asosiasi"):
            st.write("Aturan Asosiasi:")
            st.dataframe(rules)  # Menampilkan aturan asosiasi dalam bentuk tabel

        # Menampilkan kesimpulan dari aturan asosiasi
        with st.expander("Kesimpulan Aturan Asosiasi"):
            st.write("Kesimpulan Aturan Asosiasi:")
            kesimpulan_data = []
            for rule in rules.itertuples(index=False):
                antecedents = ', '.join(list(rule[0]))
                consequents = ', '.join(list(rule[1]))
                kesimpulan_data.append({'Antecedents': antecedents, 'Consequents': consequents})
                st.write(f"Jika membeli {antecedents}, maka akan membeli {consequents}")

# Definisi fungsi untuk halaman Connection Tree
def connection_tree():
    st.title("Pohon Asosiasi")

    # Cek apakah data sudah diunggah
    if 'data' not in st.session_state:
        st.write("Silakan unggah file data terlebih dahulu di halaman 'Upload Data'.")
    else:
        data = st.session_state['data']

        # Input min_support dan min_confidence
        min_support = st.slider("Minimum Support", 0.0, 1.0, 0.2, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.01)

        # Konversi data menjadi matriks biner
        encoded_data = pd.get_dummies(data)
        encoded_data = encoded_data >= 1  # Mengonversi data menjadi format biner

        # Menghitung frekuensi itemset
        frequent_itemsets = apriori(encoded_data.astype('bool'), min_support=min_support, use_colnames=True)

        # Menghitung aturan asosiasi
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # Menampilkan Connection Tree
        st.write("Visualisasi Pohon Asosiasi:")

        # Buat grafik
        G = nx.DiGraph()

        for rule in rules.itertuples(index=False):
            antecedents = ', '.join(list(rule[0]))
            consequents = ', '.join(list(rule[1]))
            G.add_edge(antecedents, consequents, weight=rule[4])  # rule[4] adalah confidence

        pos = nx.spring_layout(G)  # Posisi untuk semua simpul
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        st.pyplot(plt)

def tentang():
    st.title("Tentang SI-ASPARSUCO")

    st.markdown("""
        **SI-ASPARSUCO** (Sistem Informasi Asosiasi dengan Parameter Support dan Confident) adalah aplikasi perhitungan data mining dengan asosiasi yang memungkinkan Anda untuk:
        - Mengunggah file data (Excel atau CSV)
        - Mengatur nilai minimum confidence dan support
        - Melakukan analisis asosiasi pada data tersebut
        - Memvisualiasi analisis dalam bentuk pohon asosasi.
    """)

    st.subheader("Library python yang Digunakan")
    st.markdown("""
    - **Streamlit:** untuk frontend
    - **pandas:** untuk manipulasi data
    - **networkx:** untuk visualisasi pohon asosiasi
    - **mlxtend:** untuk perhitungan algoritma apriori
    """)

    st.subheader("Tujuan Pengembangan")
    st.markdown("""
    Aplikasi ini dikembangkan untuk memenuhi Ujian Akhir Semester Mata Kuliah Data Mining.
    """)

    st.subheader("Kelompok 6 - Asosiasi")
    st.markdown("""
    - Nasrun Nugroho
    - Febriaji Primadeni
    """)

# Membuat daftar halaman
pages = {
    "Tentang": tentang,
    "Upload Data": upload_data,
    "Analisis Asosiasi": analisis_asosiasi,
    "Pohon Asosiasi": connection_tree,
}

# Membuat sidebar untuk navigasi antar halaman dengan tombol
st.sidebar.title("Menu SI-ASPARSUCO")

# Buat tombol untuk setiap halaman
for page in pages.keys():
    if st.sidebar.button(page):
        st.session_state['current_page'] = page

# Tampilkan halaman yang dipilih
if 'current_page' in st.session_state:
    pages[st.session_state['current_page']]()
else:
    pages['Tentang']()  # Halaman default
