# Protokol YOLO–Local-HBP

## Pertanyaan penelitian

Apakah interaksi bilinear lokal pada classification branch YOLO meningkatkan
deteksi cacat kopi fine-grained tanpa mengubah regresi bounding box secara
langsung?

## Isolasi dari model klasifikasi

1. Subproyek memiliki `pyproject.toml`, namespace, config, tes, dan output sendiri.
2. Tidak ada import dari `bilinear_lmmd`.
3. HBP global Coffee-17 tidak digunakan ulang. `LocalBilinearAdapter` mempertahankan
   `H x W` dan dipasang hanya di `Detect.cv3`.
4. Dataset Coffee Defect v11 tidak digabung dengan Coffee-17.

## Tahapan

1. Audit anotasi, kelas, exact duplicate, near duplicate, dan parent Roboflow.
2. Jika perlu, buat grouped split 70/15/15 sebelum training.
3. Kunci test split.
4. Screening Y0 vs Y1 pada seed 42.
5. Hanya jika lolos, konfirmasi seed 123 dan 2026.
6. Tambahkan capacity-matched non-bilinear control sebelum klaim kontribusi.

## Metrik minimum

- mAP50-95 dan mAP50.
- Precision dan recall.
- AP per kelas dan worst-class AP.
- AP berdasarkan ukuran objek dan kepadatan objek per frame.
- Parameter, FLOPs, latency batch-1, dan FPS pada perangkat target.

## Kriteria screening

Y1 dilanjutkan bila mAP50-95 meningkat, worst-class AP tidak turun, dan tidak ada
kerusakan besar pada kelas minoritas. Angka dari halaman Roboflow tidak digunakan
sebagai baseline ilmiah; seluruh model dilatih ulang pada split hasil audit yang
sama.

