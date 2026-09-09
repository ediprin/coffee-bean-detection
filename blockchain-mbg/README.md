# Blockchain MBG SHA-256

Proyek sederhana untuk memahami penggunaan teknik kriptografi pada data transaksi dalam struktur blockchain.

## Fokus

Alur yang didemonstrasikan:

```
Data transaksi MBG
        ↓
Serialisasi deterministik
        ↓
SHA-256
        ↓
Hash transaksi
        ↓
Genesis Block
        ↓
Block 1 + previous_hash
```

Notebook juga mendemonstrasikan perubahan data transaksi dari **100 kg** menjadi **101 kg** dan menunjukkan bahwa hash SHA-256 berubah.

## Jalankan di Google Colab

[Open in Google Colab](https://colab.research.google.com/github/ediprin/coffee-bean-detection/blob/blockchain-mbg/blockchain-mbg/Blockchain_MBG_SHA256_Colab.ipynb)

## File

- `Blockchain_MBG_SHA256_Colab.ipynb` — notebook utama.
- `README.md` — dokumentasi singkat proyek.

## Catatan

Ini adalah demonstrasi blockchain sederhana untuk pembelajaran kriptografi dan integritas data. Ini belum merupakan jaringan blockchain terdistribusi lengkap.
