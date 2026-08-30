# Revisi Subbab 3.12 — Lingkungan Implementasi

## Keputusan revisi

1. Versi perangkat lunak yang digunakan pada eksperimen utama harus dipin dan diperlakukan sebagai bagian dari kontrak eksperimen. Jika Ultralytics 8.4.96 memang menjadi versi final yang digunakan, tuliskan sebagai versi eksperimen utama, bukan sekadar "versi referensi".
2. Lingkungan perangkat keras final sekurang-kurangnya mencatat CPU, GPU, RAM, sistem operasi, versi CUDA/driver, serta versi Python dan PyTorch. CPU perlu dicatat karena sebagian prapemrosesan dapat berjalan di CPU dan waktu prapemrosesan ikut dievaluasi.
3. Identitas commit yang dicatat harus mewakili keseluruhan kode eksperimen, tidak hanya kode prapemrosesan. Perubahan pada integrasi frontend, data loader, evaluasi, benchmark, atau konfigurasi pelatihan juga dapat memengaruhi hasil.
4. Manifest pembagian dataset berbasis `group_id` harus dibekukan dan dapat ditelusuri. Setiap run harus menggunakan manifest train/validation/test yang sama sesuai rancangan eksperimen.
5. Konfigurasi aktual setiap run perlu disimpan bersama hasil eksperimen, termasuk model, seed, konfigurasi prapemrosesan, parameter yang relevan seperti `m`, `gamma`, dan `T`, serta konfigurasi pelatihan.
6. Pernyataan bahwa reproducibility tidak menjamin seluruh operasi GPU identik secara bitwise dipertahankan. Yang dijamin adalah seluruh kondisi dibandingkan menggunakan prosedur reproducibility yang sama.

## Redaksi inti yang disarankan

> Implementasi penelitian menggunakan Python, PyTorch, dan Ultralytics YOLO. Versi perangkat lunak dikunci sebelum eksperimen utama dan tidak diubah selama perbandingan. Informasi versi Python, PyTorch, Ultralytics, CUDA, sistem operasi, CPU, GPU, dan perangkat keras utama dicatat.
>
> Setiap run ditautkan dengan identitas commit kode eksperimen, konfigurasi run, serta manifest pembagian dataset yang digunakan. Dengan demikian, model, seed, konfigurasi prapemrosesan, parameter pelatihan, dan pembagian data dapat ditelusuri kembali.
>
> Seed dan pengaturan reproducibility yang relevan pada perangkat lunak serta CUDA diterapkan secara konsisten pada seluruh kondisi. Penelitian tidak mengasumsikan bahwa seluruh operasi GPU menghasilkan keluaran yang identik secara bitwise, tetapi seluruh kondisi dibandingkan menggunakan prosedur reproducibility yang sama.
