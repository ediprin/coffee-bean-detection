# Proposal Artifact Workspace

Direktori ini sekarang diperlakukan sebagai **artefak proposal tesis yang dapat diedit dan direvisi langsung**. Tujuannya adalah agar keputusan yang sudah disepakati tidak hanya tersimpan di percakapan, tetapi selalu dipindahkan ke naskah proposal di repository.

## Prinsip utama

Proposal adalah **rencana penelitian**, bukan laporan hasil penelitian yang telah dijalankan. Karena itu, naskah formal Bab I–III tidak memuat hasil eksperimen penelitian sendiri, termasuk hasil pilot satu seed, D0/D0FT, historical factorization results, promotion gate, atau diagnosis hasil eksperimen.

Hasil penelitian terdahulu tetap boleh digunakan sebagai landasan masalah, teori, dan precedent metode selama sumbernya dapat diverifikasi.

## Artefak formal

```text
BAB_I_PENDAHULUAN.md
    authority Bab I saat ini

BAB_II_TINJAUAN_PUSTAKA.md
    akan menjadi authority Bab II setelah formal rewrite

BAB_III_METODOLOGI_PENELITIAN.md
    akan menjadi authority Bab III setelah formal rewrite

DAFTAR_PUSTAKA.md
    akan berisi hanya referensi yang benar-benar disitasi dalam proposal
```

`01_PROPOSAL_SKELETON.md` adalah kontrak struktur dan batas penulisan, bukan bagian yang dicetak sebagai bab.

## File kerja / backend yang masih dipertahankan

File berikut **bukan** authority naskah formal dan tidak boleh dicopy-paste mentah ke DOCX:

```text
02_BACKGROUND.md
03_PROBLEM_FORMULATION.md
04_LITERATURE_REVIEW.md
04_02_INSPECTION_QUALITY_NORMALIZED.md
04_09_RELATED_WORK_TABLE.md
05_METHODOLOGY.md
05_05_AF2_PRIMARY_SOURCE_HARDENED.md
06_RESEARCH_FLOW.md
```

Fungsinya adalah menyimpan history drafting, evidence synthesis, detail metode, dan bahan untuk formal rewrite. Git history tetap mempertahankan versi-versi sebelumnya.

## Struktur Bab I formal

Mengikuti proposal kampus acuan:

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

Tidak ada `Identifikasi Masalah`, daftar `RQ1–RQ4`, `Kontribusi yang Diharapkan`, maupun `Sistematika Penulisan` pada Bab I formal saat ini.

## Terminologi metode

Bab I menggunakan istilah:

> preprocessing citra berbasis frekuensi-angular

Pembaca tidak diasumsikan mengetahui `AF2`, nama config, branch, atau istilah internal repository. AFAB/AFAB-2 hanya disebut ketika membahas metode sumber Xu et al. Nama adaptasi penelitian, jika nantinya digunakan, harus diperkenalkan secara akademik di Bab III setelah mekanismenya dijelaskan.

## Temporal guardrail

```text
BOLEH DALAM PROPOSAL
- masalah penelitian
- hasil penelitian terdahulu yang terverifikasi
- metode yang diusulkan
- rancangan optimasi
- rancangan eksperimen
- parameter dan metrik yang akan digunakan
- evaluasi yang akan dilakukan

TIDAK BOLEH SEBAGAI HASIL PROPOSAL
- hasil seed 42
- D0 / D0FT / staged/direct results
- historical candidate results
- PROMOTE_TO_3_SEED
- classification-dominant diagnosis dari eksperimen sendiri
- nilai performa eksperimen sendiri
- klaim bahwa metode usulan telah terbukti meningkatkan kinerja
```

## Source discipline

Sebelum mengubah artefak formal:

1. klaim literatur harus berasal dari sumber paper/standar yang mendukungnya;
2. repository digunakan untuk memastikan rancangan metode yang akan dilakukan benar secara teknis;
3. fakta paper, sintesis literatur, dan rencana penelitian harus dibedakan;
4. nama internal repository tidak digunakan tanpa definisi akademik;
5. revisi penting dari percakapan harus dipindahkan ke file artefak proposal.

## Current state

```text
BAB I   = formal proposal artifact tersedia dan menjadi authority
BAB II  = scientific working source tersedia; formal proposal rewrite berikutnya
BAB III = technical source tersedia; formal proposal rewrite setelah Bab II
DOCX    = generate manual per bab setelah artefak Markdown disetujui
```

Current working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**
