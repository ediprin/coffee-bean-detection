# V1 — Dataset Primer

Status: **WORKING VARIANT — CURRENT FORMAL BASELINE**

Versi ini mempertahankan rancangan dataset yang saat ini sudah berada pada `docs/thesis/proposal/BAB_III_METODOLOGI_PENELITIAN.md`. File ini bukan duplikasi source formal; fungsinya hanya memberi identitas eksplisit bahwa metodologi aktif sekarang adalah **Versi 1 — Dataset Primer**.

## Kontrak Utama

Penelitian menggunakan dataset primer untuk deteksi objek multikelas pada biji kopi hijau. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal, sehingga:

$$
C_{target}=21.
$$

Jumlah kelas final dibekukan setelah audit kecukupan data dan sebelum pembagian dataset serta pelatihan utama.

Target pengumpulan adalah sekitar 180–220 citra sumber dengan sasaran nominal sekitar 200 citra asli. Dengan sekitar 30–50 objek per citra, sasaran nominal jumlah anotasi adalah:

$$
N_{box}\approx6.000-10.000.
$$

Setiap kelas diupayakan memiliki sekurang-kurangnya sekitar 200 objek asli, sasaran ideal sekitar 300–500 objek, dan muncul pada sekitar 30 citra sumber berbeda.

## Akuisisi dan Anotasi

- citra diambil dari atas pada latar polos dan tidak reflektif;
- posisi kamera, jarak, dan pencahayaan dikendalikan;
- biji disusun satu lapisan dengan orientasi bervariasi;
- setiap objek diberi bounding box dan label kelas;
- sumber fisik, sesi, lot/batch, dan susunan objek ditelusuri melalui metadata;
- citra yang berasal dari sumber/spesimen berkaitan diberi `group_id` yang sama;
- label yang meragukan ditinjau ulang dan validasi melibatkan pihak yang memahami penilaian mutu fisik kopi bila tersedia.

## Split dan Leakage Control

Target pembagian adalah sekitar 70% train, 15% validation, dan 15% test sebelum augmentasi. Split dilakukan berdasarkan `group_id`, sehingga kelompok sumber yang sama tidak tersebar pada lebih dari satu subset.

Data validasi digunakan untuk penghentian dini, pemilihan konfigurasi, dan analisis sensitivitas. Data uji tetap tertutup sampai `C*`, seed konfirmasi, aturan checkpoint, metrik, dan prosedur evaluasi dibekukan.

## Hubungan dengan Eksperimen

- `B0`: YOLO26n tanpa prapemrosesan tambahan;
- `B1`: CLAHE + YOLO26n;
- `B2`: `C0` + YOLO26n;
- `B3`: `C*` + YOLO26n.

`C*` dipilih pada validasi dataset primer dengan seed pengembangan 42. Konfirmasi menggunakan seed 123, 2026, dan 31415 pada dataset yang sama, kemudian dilakukan evaluasi akhir pada test set yang tetap tertutup selama pengembangan.

## Kekuatan dan Risiko

Kekuatan V1 adalah kontrol penuh terhadap akuisisi, anotasi, definisi kelas, pembentukan kelompok sumber, dan desain test set. Risiko utamanya adalah biaya pengumpulan/anotasi yang tinggi serta kemungkinan kelas langka tidak mencapai kecukupan data sebelum batas waktu tesis.

## Rule

Selama belum ada keputusan eksplisit untuk mempromosikan V2, file formal BAB III tetap mengikuti V1 ini.
