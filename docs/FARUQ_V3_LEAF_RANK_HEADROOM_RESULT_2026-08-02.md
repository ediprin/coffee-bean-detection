# Faruq-v3 Leaf-Rank Headroom Result

Tanggal: 2026-08-02

Status: validation-only, tanpa training, test tetap terkunci

Keputusan: **FAIL — hentikan global leaf reranking**

## Pertanyaan

Audit ini menguji apakah kelas daun SNI yang benar sebenarnya sudah berada dekat
peringkat teratas pada kandidat mentah YOLO26, sehingga kesalahan dapat diperbaiki
dengan reranking ringan tanpa mengubah representasi fitur.

Protokol memakai checkpoint D0 seed 42, kandidat one-to-one mentah top-500, dan
matching class-agnostic greedy pada IoU minimal 0,5. Reranking hanya lolos bila
akurasi top-1 di bawah 80%, akurasi top-3 minimal 80%, dan pemulihan top-3 terhadap
top-1 minimal 15 poin persentase.

## Hasil utama

| Metrik | Hasil |
|---|---:|
| Target validasi | 526 |
| Target memperoleh kandidat cocok | 525 |
| Proposal accessibility / matched recall | 99,81% |
| Conditional top-1 accuracy | 33,52% |
| Conditional top-2 accuracy | 41,52% |
| Conditional top-3 accuracy | 48,95% |
| Conditional top-5 accuracy | 60,38% |
| Top-3 recovery atas top-1 | +15,43 poin |
| Mean reciprocal rank | 45,86% |
| Median peringkat kelas benar | 4 |

Walaupun syarat pemulihan top-3 terpenuhi, akurasi top-3 hanya 48,95%, jauh di
bawah gate 80%. Artinya kelas benar bukan sekadar sering berada di urutan kedua
atau ketiga; pada banyak objek kelas benar sama sekali tidak kompetitif di bagian
atas distribusi skor.

## Kelas dengan headroom terendah

| Kelas | Top-1 | Top-3 | Pemulihan | Mean rank kelas benar |
|---|---:|---:|---:|---:|
| kulit_tanduk_ukuran_kecil | 4,00% | 20,00% | +16,00 poin | 16,48 |
| biji_hitam | 8,00% | 28,00% | +20,00 poin | 10,60 |
| biji_berlubang_satu | 12,00% | 40,00% | +28,00 poin | 4,52 |
| biji_muda | 12,00% | 20,00% | +8,00 poin | 6,00 |
| biji_pecah | 12,00% | 64,00% | +52,00 poin | 3,76 |
| biji_hitam_sebagian | 12,50% | 33,33% | +20,83 poin | 3,92 |
| biji_berlubang_lebih_satu | 16,67% | 54,17% | +37,50 poin | 3,46 |
| biji_normal | 26,92% | 65,38% | +38,46 poin | 3,96 |
| biji_berkulit_tanduk | 28,00% | 28,00% | +0,00 poin | 8,48 |
| biji_coklat | 32,00% | 48,00% | +16,00 poin | 10,36 |

Kelas seperti `biji_pecah`, `biji_normal`, dan dua kelas lubang masih memiliki
sebagian headroom top-3. Namun, `kulit_tanduk_ukuran_kecil`, `biji_hitam`,
`biji_muda`, `biji_berkulit_tanduk`, dan `biji_coklat` menunjukkan masalah yang
lebih dalam daripada salah urutan tipis.

## Pasangan kebingungan dominan

| Kelas benar | Prediksi | Jumlah | Kelas benar di top-3 | Median rank benar |
|---|---|---:|---:|---:|
| kulit_tanduk_ukuran_kecil | kulit_tanduk_ukuran_besar | 17 | 0,00% | 21,0 |
| biji_muda | biji_bertutul_tutul | 15 | 6,67% | 5,0 |
| biji_berlubang_satu | biji_bertutul_tutul | 14 | 42,86% | 4,5 |
| biji_berkulit_tanduk | kulit_tanduk_ukuran_besar | 9 | 0,00% | 13,0 |
| biji_berlubang_lebih_satu | biji_bertutul_tutul | 9 | 33,33% | 4,0 |
| biji_normal | kulit_tanduk_ukuran_besar | 9 | 88,89% | 2,0 |
| biji_hitam_sebagian | biji_bertutul_tutul | 8 | 37,50% | 4,0 |
| kopi_gelondong | biji_coklat | 8 | 0,00% | 5,0 |
| kulit_kopi_ukuran_besar | kulit_kopi_ukuran_sedang | 8 | 0,00% | 17,0 |
| kulit_tanduk_ukuran_sedang | kulit_tanduk_ukuran_besar | 8 | 0,00% | 19,0 |

Kebingungan ukuran kulit tanduk sangat tidak cocok untuk reranking: kelas benar
sering berada pada peringkat 17--21. Kebingungan `biji_normal` terhadap
`kulit_tanduk_ukuran_besar` lebih mungkin dipulihkan karena kelas benar hampir
selalu masih berada di top-3, tetapi pola lokal ini tidak cukup membenarkan
reranker global.

## Interpretasi

1. **Proposal bukan bottleneck pada audit kandidat mentah ini.** Hampir setiap
   target memiliki kandidat dengan IoU memadai.
2. **Global leaf reranking tidak terjustifikasi.** Top-3 di bawah 50% berarti
   mayoritas kesalahan tidak dapat diperbaiki hanya dengan menata ulang beberapa
   kelas teratas.
3. **Bottleneck kelas tersulit berada sebelum keputusan akhir**, yaitu pada
   keterpisahan representasi, observabilitas ciri, kualitas/semantik label, atau
   kombinasi ketiganya.
4. Hasil ini tidak membuktikan semua perubahan representasi mustahil. Hasil ini
   secara khusus menutup hipotesis bahwa post-hoc/global leaf reranker cukup.
5. Tidak ada dasar untuk membuka test, menambah seed, atau melatih reranker dari
   hasil ini.

## Batas perbandingan metrik

Conditional top-1 33,52% pada audit ini dihitung dari kandidat mentah top-500
yang dipasangkan secara greedy berdasarkan IoU. Angka ini **tidak boleh
dibandingkan langsung** dengan conditional top-1 sekitar 62,99% dari output
deteksi final karena populasi kandidat dan aturan matching berbeda. Keputusan
FAIL tetap sah terhadap gate audit yang telah dibekukan: top-3 mentah hanya
48,95%, bukan minimal 80%.

## Artefak

Laporan mentah Colab:
`experiments/faruq-v3-leaf-rank-headroom-v1/leaf_rank_headroom.json`
