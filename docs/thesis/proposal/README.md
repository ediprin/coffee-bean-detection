# Proposal Draft Workspace

Direktori ini berisi bahan dan naskah proposal tesis. Mulai revisi Bab I 2026-08-25, **naskah proposal formal dipisahkan tegas dari evidence eksperimen internal repository**.

## Aturan utama

Proposal adalah **rencana penelitian**. Karena itu, Bab I–III formal tidak memuat hasil eksperimen penelitian sendiri yang sudah pernah dijalankan di repository, termasuk hasil seed tunggal, D0/D0FT, historical factorization, nilai pilot, promotion gate, atau diagnosis hasil eksperimen.

Hasil penelitian terdahulu tetap boleh digunakan sebagai landasan masalah, teori, dan precedent metode selama sumber primernya terverifikasi.

## Active files

```text
01_PROPOSAL_SKELETON.md
    struktur proposal formal + proposal-only guardrails

02_BACKGROUND.md
    authoritative proposal-facing §1.1 Latar Belakang
    tanpa hasil eksperimen penelitian sendiri dan tanpa istilah AF2 yang belum diperkenalkan

03_PROBLEM_FORMULATION.md
    authoritative proposal-facing §1.2–1.5:
    Rumusan Masalah, Batasan Masalah, Tujuan Penelitian, Manfaat Penelitian

04_LITERATURE_REVIEW.md
    source-grounded Bab II working draft; masih perlu formal-language cleanup saat assembly

04_02_INSPECTION_QUALITY_NORMALIZED.md
    normalized replacement untuk §2.2

04_09_RELATED_WORK_TABLE.md
    working related-work table; masih perlu proposal-facing rewrite sebelum DOCX

05_METHODOLOGY.md
    technical methodology source; bukan teks formal yang ditempel mentah

05_05_AF2_PRIMARY_SOURCE_HARDENED.md
    technical provenance source untuk mekanisme frequency-angular;
    bukan teks formal yang ditempel mentah

06_RESEARCH_FLOW.md
    technical research-flow source; perlu proposal-facing redraw/rewrite
```

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

Pada Bab I gunakan istilah:

> preprocessing citra berbasis frekuensi-angular

Jangan mengasumsikan pembaca mengetahui istilah `AF2`. Nama/adaptasi teknis baru boleh diperkenalkan di Bab III setelah mekanisme sumber dan definisi istilah dijelaskan secara akademik.

## Temporal guardrail

```text
BOLEH DALAM PROPOSAL
- masalah penelitian
- hasil penelitian terdahulu yang terverifikasi
- metode yang diusulkan
- rancangan optimasi
- rancangan eksperimen
- metrik yang akan digunakan
- evaluasi yang akan dilakukan

TIDAK BOLEH SEBAGAI HASIL PROPOSAL
- seed-42 pilot result
- D0 / D0FT / AF2 staged result
- historical candidate result
- PROMOTE_TO_3_SEED
- classification-dominant diagnosis dari eksperimen sendiri
- nilai Macro/Bottom-3/Worst milik eksperimen sendiri
- klaim bahwa metode usulan sudah meningkatkan kinerja
```

Evidence tersebut tetap dipertahankan di foundation/protocol/result files sebagai catatan pengembangan penelitian, tetapi tidak disalin sebagai hasil ke proposal formal.

## Source discipline

Sebelum menulis naskah formal:

1. gunakan primary paper untuk klaim literatur;
2. gunakan repository hanya untuk mendefinisikan metode/rancangan yang akan dilakukan;
3. bedakan fakta paper, sintesis literatur, dan rencana penelitian;
4. jangan mengubah hasil eksperimen internal menjadi narasi hasil proposal;
5. jangan menggunakan nama internal repository tanpa definisi bagi pembaca akademik.

## Current state

```text
Bab I   = proposal-facing rewrite selesai pada source Markdown
Bab II  = working source; formal cleanup belum selesai
Bab III = technical source tersedia; formal proposal rewrite belum selesai
DOCX    = generate manual per bab setelah teks bab disetujui
```

Current working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**
