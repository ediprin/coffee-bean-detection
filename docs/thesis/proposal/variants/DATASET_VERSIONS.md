# Varian Strategi Dataset Proposal

Status: **WORKING VARIANTS — BUKAN SOURCE FORMAL**

Dokumen ini memisahkan dua alternatif rancangan dataset tanpa mengubah source formal pada `docs/thesis/proposal/`.

## Versi 1 — Dataset Primer

Source:

- `docs/thesis/proposal/variants/V1_DATASET_PRIMER.md`
- implementasi formal saat ini tetap `docs/thesis/proposal/BAB_III_METODOLOGI_PENELITIAN.md`

Karakter utama:

- dataset dikumpulkan dan dianotasi sendiri;
- target awal 20 kategori cacat/benda asing + 1 normal;
- sekitar 180–220 citra sumber dengan sasaran nominal ~200 citra;
- skala nominal sekitar 6.000–10.000 bounding box;
- split berbasis `group_id` sumber;
- kontrol terbesar terhadap akuisisi, anotasi, kelas, dan pencegahan leakage;
- biaya terbesar berada pada pengumpulan data, anotasi, validasi label, dan kecukupan kelas langka.

## Versi 2 — Multi-Dataset Publik

Source:

- `docs/thesis/proposal/variants/V2_MULTI_DATASET_PUBLIK.md`

Karakter utama:

- tidak melakukan pengumpulan dataset primer sebagai sumber utama;
- menggunakan beberapa dataset publik deteksi cacat biji kopi hijau yang lolos audit provenance, lisensi, anotasi, dan duplikasi;
- dataset **tidak langsung digabung menjadi satu label space**;
- setiap dataset diperlakukan sebagai benchmark independen dengan kelasnya sendiri;
- satu dataset pengembangan digunakan untuk memilih konfigurasi `C*`;
- `C*` kemudian dibekukan dan diuji pada dataset publik lain sebagai konfirmasi lintas dataset;
- kontribusi utama dinilai dari konsistensi delta `B3-B0` lintas dataset dan seed.

## Perbedaan Inti

| Aspek | V1 Dataset Primer | V2 Multi-Dataset Publik |
|---|---|---|
| Sumber data | Dikumpulkan sendiri | Beberapa dataset publik |
| Kontrol akuisisi | Tinggi | Terbatas pada data yang tersedia |
| Kontrol anotasi | Tinggi | Perlu audit dan koreksi bila diizinkan |
| Taksonomi | Dapat dirancang dari awal | Berbeda antar-dataset |
| Risiko leakage | Dikendalikan melalui `group_id` saat akuisisi | Perlu audit hash, pHash, versi, fork, dan augmentasi |
| Risiko duplikasi | Relatif rendah | Tinggi karena dataset publik dapat merupakan fork/derivatif |
| Kebutuhan harmonisasi kelas | Rendah | Tinggi jika dipaksa merge; karena itu tidak dilakukan pada analisis utama |
| Uji generalisasi | Terbatas pada satu domain akuisisi | Lebih kuat karena lintas dataset/domain |
| Beban kerja | Akuisisi + anotasi tinggi | Audit provenance + data engineering tinggi |
| Klaim yang paling kuat | Kontrol eksperimental pada satu dataset primer | Konsistensi efek metode pada beberapa dataset independen |

## Aturan Source-of-Truth

1. Selama belum ada keputusan eksplisit, **V1 tetap source formal aktif**.
2. File pada folder `variants/` tidak masuk build proposal formal dan tidak mengubah citation gate.
3. Jika V2 dipilih, BAB I, BAB II, BAB III, Gambar 3.1, bibliography, dan audit formal harus disinkronkan sebagai satu perubahan terkontrol.
4. Dataset publik tidak boleh dinyatakan independen hanya karena memiliki nama atau workspace berbeda. Provenance dan kemiripan isi wajib diaudit terlebih dahulu.
5. Dataset klasifikasi tanpa bounding box tidak masuk benchmark utama deteksi kecuali terdapat prosedur konversi yang dapat dipertanggungjawabkan dan dibekukan sebelum eksperimen.
