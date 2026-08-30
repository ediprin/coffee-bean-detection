# Revisi 3.11 — Evaluasi Efisiensi Komputasi

Catatan ini merekam keputusan revisi Subbab 3.11 sebelum perubahan diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## Keputusan utama

1. Efisiensi tetap dievaluasi pada tingkat sistem karena CLAHE dan prapemrosesan frekuensi-angular tidak menambah parameter terlatih tetapi tetap menambah biaya komputasi.
2. Benchmark utama tetap menggunakan ukuran masukan 640 × 640, batch 1, perangkat yang sama, dan presisi komputasi yang sama.
3. Tetap laporkan tiga komponen waktu: `t_pra`, `t_model`, dan `t_total`, tetapi `t_total` harus diukur langsung secara end-to-end, bukan hanya dihitung dari penjumlahan dua benchmark terpisah.
4. Seluruh overhead yang hanya diperlukan oleh suatu metode harus masuk ke biaya metode tersebut, termasuk konversi dtype/ruang warna, perpindahan CPU–GPU, atau konversi representasi lain yang diperlukan oleh frontend. Disk I/O umum yang identik bagi semua kondisi tidak dimasukkan ke latency utama.
5. Untuk kondisi tanpa prapemrosesan tambahan, `t_pra` diperlakukan sebagai nol pada dekomposisi biaya metode.
6. Jumlah warm-up dan jumlah pengulangan benchmark ditetapkan sebelum perbandingan dan dibuat sama pada seluruh kondisi. Angka konkretnya diisi setelah pipeline benchmark final diuji, bukan dipilih berdasarkan hasil.
7. Operasi GPU disinkronkan pada batas pengukuran. Jika sebagian frontend berjalan di CPU, total pipeline diukur menggunakan wall-clock timing yang mencakup pekerjaan CPU dan sinkronisasi GPU yang relevan; jangan mengandalkan CUDA event saja untuk seluruh pipeline.
8. Median end-to-end latency batch 1 digunakan sebagai ukuran utama efisiensi dan sebagai tie-break pada Subbab 3.6.2. Variasi latency, misalnya IQR atau statistik konsisten lain yang ditetapkan sebelum benchmark, turut dilaporkan.
9. Throughput/FPS harus berasal dari protokol yang sama. Untuk batch 1, throughput dapat diturunkan dari latency total dengan definisi yang dinyatakan jelas. Jangan mencampur FPS batch besar dengan latency batch 1 tanpa penjelasan.
10. Jumlah parameter model tetap dilaporkan untuk menunjukkan bahwa perbedaan pada kondisi YOLO26n tidak berasal dari penambahan parameter trainable. `Peak allocated GPU memory` dilaporkan secara presisi sebagai penggunaan memori GPU puncak, bukan sebagai total memori sistem.
11. Jika analisis RT-DETR dilakukan, efisiensi RT-DETR dilaporkan terpisah sebagai analisis tambahan dan tidak dijadikan dasar perbandingan efisiensi utama antar kondisi YOLO26n.

## Redaksi inti yang disarankan

> Efisiensi dievaluasi pada tingkat sistem karena prapemrosesan tidak menambah parameter terlatih tetapi tetap menambah biaya komputasi. Pengukuran utama menggunakan masukan 640 × 640 dan batch 1 pada perangkat serta presisi komputasi yang sama.
>
> Untuk kondisi dengan prapemrosesan, dilaporkan waktu prapemrosesan `t_pra`, waktu inferensi model `t_model`, dan latency total end-to-end `t_total`. Seluruh operasi tambahan yang hanya diperlukan oleh suatu metode, termasuk konversi representasi atau perpindahan perangkat yang diperlukan preprocessing, dimasukkan ke biaya metode tersebut. Disk I/O yang identik untuk seluruh kondisi tidak dimasukkan.
>
> Benchmark menggunakan jumlah warm-up dan pengulangan yang sama dan ditetapkan sebelum perbandingan. Operasi GPU disinkronkan pada batas pengukuran. Median latency total digunakan sebagai ukuran utama efisiensi dan sebagai tie-break pada Subbab 3.6.2, sedangkan variasi latency turut dilaporkan. Jumlah parameter dan peak allocated GPU memory dilaporkan sebagai informasi tambahan.
