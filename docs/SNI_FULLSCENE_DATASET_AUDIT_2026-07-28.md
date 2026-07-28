# Audit Full-Scene Dataset SNI

Tanggal audit: 28 Juli 2026

## Ruang lingkup

Audit ini memeriksa dua ekspor Roboflow asli yang diunduh tanpa menjalankan
training:

- `YOLO SKRIPSI 2` (`adrian_detection`, COCO detection);
- `robusta_SNI_Dataset` (`faruq_segmentation`, COCO segmentation).

Arsip memiliki lisensi yang tertulis sebagai CC BY 4.0. Temuan di bawah berlaku
untuk versi ekspor yang dicatat dalam
`coffee-sni-detection-fullscene-v1/complete.json`, bukan untuk versi Roboflow
lain.

## Ringkasan data

| Dataset | Split arsip | Gambar | Anotasi | Objek/gambar |
|---|---|---:|---:|---:|
| Adrian | train | 4.900 | 4.897 | median 1, maksimum 1 |
| Adrian | valid | 1.400 | 1.400 | median 1, maksimum 2 |
| Adrian | test | 700 | 21.053 | median 35, rentang 21–36 |
| Faruq | train | 1.677 | 2.979 | median 1, maksimum 6 |
| Faruq | valid | 284 | 533 | median 1, maksimum 5 |
| Faruq | test | 187 | 346 | median 1, maksimum 6 |

Totalnya adalah 9.148 gambar dan 31.208 anotasi. Semua file gambar yang
direferensikan COCO tersedia.

## Taksonomi

Kedua file COCO memiliki satu category header kosong yang tidak dipakai:
`coffee-E6YO` pada Adrian dan `robusta-SNI-Dataset` pada Faruq. Category tersebut
bukan kelas objek.

Faruq memiliki 21 kelas objek yang dapat dipetakan langsung ke taksonomi
kanonis, termasuk `biji_normal`. Adrian memiliki 27 kelas objek karena batu,
tanah, dan ranting dipisahkan. Untuk eksperimen 21 kelas, ketiganya harus
digabung berdasarkan ukuran:

| Kelas kanonis | Kelas Adrian yang digabung |
|---|---|
| `tanah_batu_ranting_besar` | Batu besar, Tanah besar, Ranting besar |
| `tanah_batu_ranting_sedang` | Batu sedang, Tanah sedang, Ranting sedang |
| `tanah_batu_ranting_kecil` | Batu kecil, Tanah kecil, Ranting kecil |

Delapan belas kelas lainnya memiliki padanan langsung. Mapping ini adalah
harmonisasi label deteksi, bukan penerapan rumus grading SNI.

## Temuan kritis

### 1. Split bawaan mengalami leakage

Hash SHA-256 atas 9.148 file menghasilkan 9.080 gambar unik dan 68 grup
duplikat. Enam belas grup duplikat identik menyeberang split di dataset Adrian.
Tidak ditemukan gambar identik lintas Adrian dan Faruq.

Rekonstruksi parent dari nama sebelum suffix `.rf.<hash>` menemukan:

- Adrian: 81 parent muncul pada train dan valid.
- Adrian test: 700 file hanya berasal dari 43 nama parent; evaluasi atas seluruh
  700 file akan mengalami pseudo-replikasi.
- Faruq: 48 parent menyeberang train–test, 72 train–valid, dan 9 valid–test.
  Secara union terdapat 123 parent yang muncul pada lebih dari satu split.

Karena itu, train/valid/test bawaan tidak aman untuk klaim ilmiah.

### 2. Adrian mencampur dua rezim citra

Train dan valid Adrian hampir seluruhnya berisi satu objek, sedangkan test
berisi 21–36 objek. Ini bukan sekadar perbedaan split, tetapi perubahan domain
dari single-object ke multi-object. Test juga hanya memiliki 43 parent.

Adrian test berguna sebagai sumber scene multiobjek dan evaluasi domain
terpisah setelah regrouping. Ia belum merupakan scene 300 g: maksimum hanya 36
objek, jauh di bawah target ratusan biji.

### 3. Faruq memiliki polygon berguna, tetapi metadata area salah

Seluruh 3.858 anotasi Faruq memiliki polygon. Polygon bukan sekadar rectangle:
median 26 vertex dan rasio median luas polygon terhadap bbox sekitar 0,74.
Mask ini layak menjadi sumber cutout setelah audit visual.

Namun field COCO `area` sama dengan luas bbox, bukan luas polygon. Nilai tersebut
harus dihitung ulang sebelum evaluasi instance segmentation atau stratifikasi
ukuran yang menggunakan `area`.

### 4. Ada noise label dan geometri

Audit crop yang sudah tersimpan pada `coffee-sni-instance-crop-v1/audit.json`
menemukan:

- 47 grup crop duplikat sebelum resolusi;
- 41 grup crop identik dengan label bertentangan;
- 127 crop konflik dikarantina;
- 7 crop duplikat label-sama dibuang.

Konflik paling banyak berasal dari Faruq. Ini adalah bukti label noise yang
harus diaudit manual, bukan alasan memilih salah satu label secara otomatis.

Adrian test juga memiliki delapan bbox dengan koordinat `-0.01` pada tepi
gambar. Kesalahan ini kecil dan dapat di-clamp, tetapi harus dikoreksi saat
materialisasi dataset.

### 5. Independensi kedua proyek belum sepenuhnya terbukti

Tidak ditemukan:

- hash gambar identik lintas Adrian–Faruq;
- grup crop identik lintas dataset pada audit crop sebelumnya;
- pasangan objek lintas dataset pada pemeriksaan object-crop visual hash yang
  sangat ketat.

Temuan ini mendukung bahwa keduanya bukan salinan byte-identik. Namun latar
putih dan objek homogen membuat perceptual hash tingkat gambar menghasilkan
banyak false positive. Karena provenance lot, kamera, dan sesi tidak tersedia,
keduanya belum boleh disebut dua domain independen hanya dari audit ini.

## Keputusan penggunaan

### Tidak boleh dilakukan

- Menggabungkan split Roboflow apa adanya.
- Menganggap 7.000 + 2.148 file sebagai 9.148 sampel independen.
- Menggunakan semua 700 turunan Adrian test sebagai unit statistik independen.
- Menyebut Faruq atau Adrian sebagai real dense 300 g benchmark.
- Memakai data sintetis sebagai validation atau test dunia nyata.

### Peran yang layak

| Sumber | Peran yang diperbolehkan |
|---|---|
| Faruq polygon | Sumber mask/cutout train dan domain sparse-segmentation |
| Adrian single-object | Sumber tambahan train setelah regrouping dan audit label |
| Adrian multi-object | Pilot deteksi multiobjek, dipisah berdasarkan 43 parent |
| VA-DCP 300 g | Augmentasi train untuk density/visibility, bukan pengganti test nyata |

## Protokol pemulihan dataset

1. Harmonisasikan semua anotasi ke 21 kelas kanonis.
2. Abaikan category header yang tidak memiliki instance.
3. Clamp delapan bbox Adrian dan hitung ulang area polygon Faruq.
4. Karantina konflik crop/label; jangan memutuskan label dari nama file saja.
5. Bangun grup sumber dari parent Roboflow, exact hash, dan klaster augmentasi
   yang diverifikasi. Seluruh saudara masuk split yang sama.
6. Untuk evaluasi, pilih satu representasi kanonis per parent augmentasi atau
   gunakan agregasi per parent agar turunan tidak memperbesar ukuran sampel.
7. Buat grouped split baru sebelum augmentasi apa pun.
8. Laporkan hasil Faruq-sparse dan Adrian-multiobject secara terpisah; pooled
   score hanya pelengkap.
9. Gunakan A0 real-only, A1 empirical copy-paste, dan A2 VA-DCP dengan detector
   serta hyperparameter yang sama. A1/A2 hanya menambah train.
10. Kunci test real. Klaim target 300 g tetap memerlukan foto 300 g nyata dengan
    anotasi instance dan visibility.

## Putusan

Dataset **layak diselamatkan untuk training dan pilot deteksi 21 kelas**, tetapi
**tidak siap training dalam bentuk ekspor saat ini**. Langkah berikutnya adalah
materialisasi unified 21-class grouped dataset beserta quarantine report.
VA-DCP baru boleh ditambahkan setelah dataset real tersebut lolos audit leakage
dan label.

## Hasil materialisasi terverifikasi

Materializer kemudian diimplementasikan sebagai:

```bash
python -u -m coffee_detector.prepare_sni_fullscene \
  --adrian-root /path/to/adrian_detection \
  --faruq-root /path/to/faruq_segmentation \
  --crop-manifest /path/to/coffee-sni-instance-crop-v1/manifest.csv \
  --output-root /path/to/sni21-fullscene-v1 \
  --seed 42
```

Split detector memakai authority `generated_split` dari manifest crop yang
sama. Dengan demikian, crop train yang dipakai VA-DCP tidak berasal dari
full-scene validation atau test.

Uji materialisasi penuh pada 28 Juli 2026 menghasilkan:

| Split baru | Gambar | Anotasi |
|---|---:|---:|
| train | 8.011 | 20.959 |
| validation | 416 | 4.969 |
| test | 451 | 4.832 |

Audit pascamaterialisasi:

- 21 kelas tersedia pada setiap split;
- 7.652 identity group dan 0 group lintas split;
- 270 gambar dikarantina;
- 136 gambar berasal dari exact-image annotation conflict;
- 134 gambar memuat anotasi yang sudah dikeluarkan audit crop;
- 1.638 gambar Faruq diputar 90 derajat searah jarum jam agar piksel sesuai
  koordinat COCO;
- 116 bbox di-clamp, seluruh koreksi hanya 0,01 piksel;
- 0 error label, 0 konflik anotasi output, dan 0 duplicate component lintas
  split;
- status akhir `TRAINING_READY=True`.

Jumlah 270 quarantine adalah 2,95% dari 9.148 gambar sumber. Kebijakan
quarantine sengaja konservatif: bila satu objek dalam suatu scene memiliki
konflik, seluruh scene dikeluarkan agar objek tersebut tidak berubah menjadi
false background.

Materialisasi ini belum menjalankan training. Test tetap terkunci.

## Audit visual pascamaterialisasi

Audit visual ground-truth dijalankan hanya pada split train dan validation:

```bash
python -u -m coffee_detector.run_sni_fullscene_visual_audit \
  --data-root /path/to/sni21-fullscene-v1 \
  --output-root /path/to/sni21-fullscene-visual-audit \
  --dense-samples 12 \
  --rotated-samples 12 \
  --seed 42
```

Hasilnya:

- 8.427 gambar dan 25.928 bbox dipindai;
- 21/21 kelas memiliki contoh visual;
- bbox pada scene padat mengikuti objek;
- sampel Faruq yang diperbaiki orientasinya memiliki bbox yang selaras;
- tidak ada data test yang dirender;
- training tetap belum dijalankan.

Tiga contact sheet yang dihasilkan mencakup scene terpadat, sampel Faruq yang
diputar, dan satu contoh untuk setiap kelas. Audit visual menyatakan output
materialisasi layak dipakai untuk eksperimen terkontrol.

Putusan ini tidak mengubah batas klaim: dataset nyata mencampur scene satu
objek dan scene multiobjek dengan sekitar 35 anotasi, tetapi belum menyediakan
benchmark nyata 300 g. VA-DCP dapat dipakai untuk menambah kepadatan dan
oklusi pada train, sedangkan validasi dan test harus tetap memakai gambar
nyata. Klaim performa pada scene 300 g tetap memerlukan test nyata 300 g yang
dianotasi terpisah.
