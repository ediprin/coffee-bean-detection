# Proposal Artifact Workspace

Direktori ini adalah **source-of-truth naskah proposal tesis**. Setiap revisi substantif yang disepakati harus dipindahkan ke artefak formal di repository agar keputusan tidak hanya tersimpan di percakapan.

## Prinsip utama

Proposal adalah **rencana penelitian**, bukan laporan hasil penelitian yang telah dijalankan. Karena itu, Bab I–III formal tidak memuat hasil eksperimen penelitian sendiri, termasuk hasil pilot satu seed, D0/D0FT, historical factorization results, promotion gate, atau diagnosis pasca-eksperimen.

Hasil penelitian terdahulu boleh digunakan sebagai landasan masalah, teori, dan metode **hanya jika klaim dan metadata sumbernya dapat diverifikasi**.

## Hard gate sitasi resmi

Sitasi merupakan **hard gate** proposal. `docs/thesis/sources/OFFICIAL_CITATION_AUDIT.md` menjadi authority untuk menentukan apakah sebuah referensi sudah aman dipromosikan ke daftar pustaka formal.

Aturannya:

1. halaman resmi penerbit/proceedings/badan standar atau primary preprint digunakan sebagai otoritas metadata;
2. master reference workbook, `CANONICAL_SOURCE_KEYS.md`, DBLP, Crossref, Google Scholar, Semantic Scholar, ResearchGate, dan sumber agregator hanya menjadi locator/corroboration;
3. DOI, author, title, venue, volume, issue, pages/article number, quartile, dan indexing **tidak boleh ditebak**;
4. metadata resmi tidak otomatis membuktikan klaim metodologis — klaim teknis tetap harus ditelusuri ke full text primer;
5. sumber yang masih `PENDING` tidak boleh dianggap citation-ready;
6. `DAFTAR_PUSTAKA.md` diblokir sampai audit resmi terhadap seluruh sitasi yang dipakai pada Bab I–III selesai.

## Artefak formal utama

```text
BAB_I_PENDAHULUAN.md
    authority Bab I

BAB_II_TINJAUAN_PUSTAKA.md
    authority Bab II

BAB_III_METODOLOGI_PENELITIAN.md
    authority Bab III

DAFTAR_PUSTAKA.md
    BELUM DIBANGUN — blocked by OFFICIAL_CITATION_AUDIT.md
```

`01_PROPOSAL_SKELETON.md` adalah kontrak struktur dan batas penulisan, bukan bagian yang dicetak sebagai bab.

## File backend / legacy drafting

File berikut bukan authority naskah formal dan tidak boleh dicopy-paste mentah ke DOCX:

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

Fungsinya adalah menyimpan riwayat drafting, evidence synthesis, detail teknis, dan provenance metode. Git history tetap mempertahankan versi-versi sebelumnya.

## Aturan terminologi

Bab I menggunakan istilah **preprocessing citra berbasis frekuensi-angular** tanpa mengasumsikan pembaca mengetahui `AF2`. Bab II menggunakan nama metode asli ketika membahas literatur, misalnya AFAB/AFAB-2 pada Xu et al. Nama konfigurasi internal repository tidak digunakan sebagai istilah akademik.

Bab III menjelaskan adaptasi mekanisme secara akademik. Nama pendek implementasi hanya boleh dipakai jika didefinisikan secara eksplisit setelah asal mekanismenya dijelaskan.

## Temporal guardrail

```text
BOLEH DALAM PROPOSAL
- masalah penelitian
- hasil penelitian terdahulu yang terverifikasi
- teori dan metode terdahulu yang terverifikasi
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

1. Klaim literatur harus berasal dari paper/standar primer yang benar-benar mendukungnya.
2. Repository dipakai untuk memastikan rancangan metode yang akan dilakukan benar secara teknis.
3. Fakta paper, sintesis literatur, dan rencana penelitian harus dibedakan.
4. Nama internal repository tidak digunakan tanpa definisi akademik.
5. Artefak formal harus dapat dibaca oleh dosen/penguji tanpa mengetahui repository.
6. Hasil eksperimen internal tidak diubah menjadi narasi hasil proposal.
7. Tidak ada entri bibliografi final yang dibuat dari ingatan atau metadata yang belum lolos official-source gate.

## Current state

```text
BAB I       = formal proposal artifact tersedia
BAB II      = formal proposal-facing rewrite tersedia; citation audit berlangsung
BAB III     = formal proposal-facing rewrite tersedia; citation audit berlangsung
REFERENSI   = OFFICIAL_CITATION_AUDIT.md aktif; DAFTAR_PUSTAKA.md diblokir
DOCX        = belum digenerate; tunggu audit teks + sitasi formal
```

Current working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**
