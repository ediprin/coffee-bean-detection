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

## Versi 2 — Dataset Utama + Konfirmasi Multi-Dataset

Source:

- `docs/thesis/proposal/variants/V2_MULTI_DATASET_PUBLIK.md`

Karakter utama:

- **robusta_SNI_Dataset** digunakan sebagai dataset utama untuk pengembangan dan pemilihan konfigurasi `C*`;
- **Coffee Bean Defect (Capstone)**, **Green Coffee Bean Defects (Lulus)**, dan **Coffee Bean Defects (Niacubilla)** digunakan sebagai dataset konfirmasi;
- setiap dataset digunakan secara terpisah dengan kelasnya masing-masing dan tidak digabung menjadi satu dataset;
- `C*` dipilih hanya pada robusta_SNI_Dataset, kemudian dibekukan;
- pada dataset konfirmasi hanya dibandingkan baseline YOLO26n (`B0`) dengan konfigurasi final (`B3`);
- model dilatih kembali pada masing-masing dataset dari bobot awal resmi yang sama;
- hasil lintas dataset digunakan untuk menilai konsistensi pengaruh metode, bukan generalisasi antarvarietas kopi.

## Perbedaan Inti

| Aspek | V1 Dataset Primer | V2 Dataset Utama + Konfirmasi |
|---|---|---|
| Sumber data | Dikumpulkan sendiri | Dataset yang telah tersedia |
| Dataset pengembangan | Dataset primer | robusta_SNI_Dataset |
| Dataset konfirmasi | Tidak ada | Capstone, Lulus, Niacubilla |
| Taksonomi | Dirancang dari awal | Berbeda antar-dataset |
| Penggabungan dataset | Tidak relevan | Tidak dilakukan |
| Pemilihan `C*` | Pada dataset primer | Hanya pada robusta_SNI_Dataset |
| Uji tambahan | Beberapa seed pada satu dataset | Baseline vs `C*` pada tiga dataset lain |
| Klaim utama | Efek metode pada dataset primer | Efek metode pada dataset utama dan konsistensinya pada dataset lain |

## Aturan Source-of-Truth

1. Selama belum ada keputusan eksplisit, **V1 tetap source formal aktif**.
2. File pada folder `variants/` tidak masuk build proposal formal dan tidak mengubah citation gate.
3. Jika V2 dipilih, BAB I, BAB II, BAB III, Gambar 3.1, bibliography, dan audit formal harus disinkronkan sebagai satu perubahan terkontrol.
4. Dataset konfirmasi harus diperiksa sebelum digunakan agar versi, anotasi, pembagian data, dan sumbernya jelas.
5. Dataset klasifikasi tanpa bounding box tidak masuk benchmark utama deteksi kecuali terdapat prosedur konversi yang dapat dipertanggungjawabkan sebelum eksperimen.
