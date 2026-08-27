# Faruq-v3 AF2 Box-Score Factorial Protocol

Tanggal pembekuan: 27 Agustus 2026.

## Pertanyaan penelitian

Diagnosis tiga seed menunjukkan bahwa AF2 tidak meningkatkan aksesibilitas
proposal mentah, tetapi meningkatkan keputusan klasifikasi dan retensi proposal
akhir. Studi ini menguji apakah keuntungan AF2 dapat dipertahankan ketika
regresi box berasal dari D0FT, sehingga arah arsitektur berikutnya tidak lagi
ditentukan dengan menebak modul.

## Desain terkunci

Eksperimen memakai checkpoint seed 42 yang optimization-matched:

- `D0FT`: kontrol detector setelah jadwal fine-tuning yang sama;
- `AF2`: kandidat AFAB-2 parameter-free yang telah lolos konfirmasi tiga seed.

Tensor raw branch `one2one` ditukar pada indeks anchor/grid yang sama, sebelum
decode dan post-processing:

| Arm | Sumber regresi box | Sumber class score |
|---|---|---|
| `DD` | D0FT | D0FT |
| `DA` | D0FT | AF2 |
| `AD` | AF2 | D0FT |
| `AA` | AF2 | AF2 |

Tidak ada weight averaging, training, ROI/crop kedua, decoded-box dependency,
akses test, atau pemilihan hyperparameter berdasarkan validation.

## Gate validitas

Hasil hybrid hanya boleh ditafsirkan jika:

1. kedua checkpoint memiliki ontologi 21 kelas, stride, dan shape raw anchor
   yang identik;
2. post-processing custom untuk `DD` dan `AA` identik dengan output native pada
   setiap gambar;
3. Macro, Bottom-3, dan Worst `DD`/`AA` berada maksimal 0,2 poin dari hasil
   historis validation seed 42;
4. seluruh 21 kelas tersedia pada validation;
5. training dan test tetap tidak diakses.

Jika gate kalibrasi gagal, keputusan adalah
`INVALID_EVALUATOR_OR_ALIGNMENT`; angka hybrid tidak boleh dipakai.

## Keputusan mekanistik

`SUPPORT_D0FT_BOX_AF2_SCORE_ARCHITECTURE` hanya jika `DA` memenuhi semuanya:

- Macro maksimal 0,1 poin di bawah `AA`;
- Bottom-3 tidak lebih rendah dari `AA`;
- Worst-class tidak lebih rendah dari `AA`.

Jika ketiga metrik `DA` lebih rendah dari `AA`, simpulan menjadi
`AF2_BOX_SCORE_INTERACTION_NECESSARY`. Kondisi lainnya dilaporkan sebagai
`MIXED_BOX_SCORE_INTERACTION` dan wajib dianalisis per kelas sebelum perubahan
arsitektur.

Studi ini bersifat diagnosis validation-only. Hasil positif hanya memberi
otorisasi untuk mengimplementasikan cabang klasifikasi AF2 yang terpisah; hasil
ini bukan klaim performa model baru.

## Artefak

- Output root: `experiments/faruq-v3-af2-box-score-factorial-v1`
- Summary: `af2_box_score_factorial.json`
- Notebook: `notebooks/Faruq_V3_AF2_Box_Score_Factorial_Colab.ipynb`

