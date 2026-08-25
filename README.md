# Proposal Tesis — Source of Truth

Branch `proposal/thesis-foundation` dikhususkan untuk artefak proposal tesis.

## Satu-satunya naskah formal

Semua agent, revisi, audit, dan proses build proposal **harus menggunakan direktori berikut sebagai sumber utama**:

```text
docs/thesis/proposal/
├── 01_PROPOSAL_SKELETON.md
├── BAB_I_PENDAHULUAN.md
├── BAB_II_TINJAUAN_PUSTAKA.md
├── BAB_III_METODOLOGI_PENELITIAN.md
└── DAFTAR_PUSTAKA.md
```

Salinan BAB I–III yang sebelumnya berada di root repository telah dihapus untuk mencegah dua sumber naskah yang berbeda.

## Aturan kerja

- Proposal adalah rencana penelitian, bukan laporan hasil eksperimen.
- BAB I–III tidak boleh memuat hasil eksperimen penelitian sendiri, hasil pilot, nilai mAP internal, hasil per-seed internal, promotion gate, atau diagnosis yang diperoleh setelah eksperimen.
- **Subbab 1.2 Rumusan Masalah dan 1.4 Tujuan Penelitian berstatus LOCKED** dan tidak boleh diubah tanpa perintah eksplisit pengguna.
- Klaim literatur harus mengikuti paper, standar, atau sumber resmi yang benar-benar mendukung klaim tersebut.
- Metadata bibliografis seperti nama penulis, judul, DOI, volume, nomor artikel, dan halaman tidak boleh ditebak.
- Hanya referensi yang benar-benar disitasi dalam BAB I–III formal yang dimasukkan ke `docs/thesis/proposal/DAFTAR_PUSTAKA.md`.
- Backend audit sumber berada di `docs/thesis/sources/` dan digunakan untuk memverifikasi sitasi, bukan sebagai pengganti naskah proposal.
- Istilah internal eksperimen seperti nama branch atau nama konfigurasi historis tidak digunakan dalam naskah formal. Variasi metode dijelaskan melalui faktor akademiknya.
- Setiap perubahan substansial pada metodologi harus diperiksa kembali konsistensinya terhadap BAB I, BAB II, BAB III, dan daftar pustaka.
- Generator DOCX hanya membaca naskah formal dari `docs/thesis/proposal/`.

## Rancangan yang berlaku saat ini

- dataset utama: dataset primer yang akan dikumpulkan;
- target awal kelas: 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 + 1 kelas biji normal, dengan jumlah kelas akhir mengikuti kecukupan data;
- model utama: YOLO26n;
- pembanding utama: YOLO26n tanpa prapemrosesan dan CLAHE + YOLO26n;
- metode: prapemrosesan frekuensi-angular dengan konfigurasi referensi dan pengujian variasi desain;
- variasi desain utama: fungsi jendela, orientasi tak bertanda dengan resolusi sudut tetap, tiga pita radial, ambang lunak, dan panduan luminansi;
- evaluasi lintas arsitektur menggunakan RT-DETRv3-R18 bersifat opsional;
- metrik utama: mAP50–95.

## Judul kerja

**Analisis dan Optimasi Prapemrosesan Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**
