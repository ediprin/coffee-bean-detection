# Build Proposal Tesis ke DOCX

Dokumen proposal formal dibangun otomatis dari **satu-satunya source of truth** pada branch `proposal/thesis-foundation`, yaitu:

```text
docs/thesis/proposal/
├── BAB_I_PENDAHULUAN.md
├── BAB_II_TINJAUAN_PUSTAKA.md
├── BAB_III_METODOLOGI_PENELITIAN.md
└── DAFTAR_PUSTAKA.md
```

File BAB duplikat di root repository tidak digunakan dan telah dihapus. Generator berada di `tools/thesis_docx/build_proposal.py` dan workflow GitHub Actions berada di `.github/workflows/build-proposal-docx.yml`.

## Format yang diterapkan

Generator menerapkan kontrak format utama Sekolah Pascasarjana USU yang digunakan pada proposal ini:

- kertas A4;
- Times New Roman 12 pt;
- margin kiri 4 cm, atas 3 cm, kanan 3 cm, bawah 3 cm;
- paragraf isi rata kiri-kanan, spasi 1,5, indent baris pertama 1,27 cm;
- judul BAB kapital, tebal, rata tengah, dan dimulai pada halaman baru;
- subbab/anak subbab tebal dan rata kiri;
- tabel rata tengah, caption di atas, spasi tunggal;
- persamaan dipertahankan sebagai native Word equation (OMML), rata tengah, dengan nomor persamaan berbasis bab di sisi kanan;
- bagian awal menggunakan nomor halaman Romawi kecil; bagian utama menggunakan angka Arab;
- halaman pembuka bagian utama menampilkan nomor halaman di kanan bawah, sedangkan halaman lanjutan di kanan atas;
- daftar pustaka menggunakan spasi tunggal dengan hanging indent;
- Daftar Isi, Daftar Tabel, dan Daftar Gambar dibuat sebagai field Word agar dapat diperbarui dari struktur dokumen.

## Build otomatis

Setiap push ke branch `proposal/thesis-foundation` yang mengubah isi `docs/thesis/proposal/`, generator, README, panduan build, atau workflow akan menjalankan workflow **Build Proposal DOCX**.

Workflow terlebih dahulu memverifikasi keberadaan keempat sumber formal di `docs/thesis/proposal/`. Generator kemudian membaca langsung file tersebut; tidak ada mekanisme fallback ke naskah root.

Hasil build disimpan sebagai GitHub Actions artifact bernama:

```text
proposal-thesis-usu
```

dengan isi:

```text
Proposal_Tesis_USU.docx
```

Artifact tidak perlu di-commit ke repository karena dapat dibangun ulang dari Markdown pada setiap revisi.

## Build manual dari GitHub Actions

Buka:

```text
Actions → Build Proposal DOCX → Run workflow
```

Input metadata sampul bersifat opsional:

- nama mahasiswa;
- NIM / singkatan program studi;
- nama program studi;
- tahun;
- label dokumen pada sampul.

Jika input manual dikosongkan, workflow akan mencoba menggunakan repository variables berikut:

```text
THESIS_STUDENT
THESIS_NIM
THESIS_PRODI
THESIS_YEAR
THESIS_LABEL
```

Jika repository variables juga belum diisi, generator menggunakan placeholder yang jelas sehingga build tetap berhasil dan tidak menebak identitas mahasiswa.

## Build lokal

Diperlukan Python, Pandoc, dan dependensi Python pada `requirements.txt`.

Dari root repository, jalankan:

```bash
pip install -r tools/thesis_docx/requirements.txt
python tools/thesis_docx/build_proposal.py \
  --repo . \
  --output build/Proposal_Tesis_USU.docx
```

Argumen `--repo .` menunjuk root repository agar generator dapat menemukan `docs/thesis/proposal/` dan README. Generator **tidak** membaca BAB I–III dari root.

Metadata sampul dapat diberikan melalui argumen:

```bash
python tools/thesis_docx/build_proposal.py \
  --repo . \
  --output build/Proposal_Tesis_USU.docx \
  --student "NAMA MAHASISWA" \
  --nim "NIM / SINGKATAN PRODI" \
  --prodi "NAMA PROGRAM STUDI" \
  --year "2026" \
  --label "TESIS"
```

atau melalui environment variables dengan nama yang sama seperti repository variables di atas.

## Penting sebelum pengumpulan

Daftar Isi, Daftar Tabel, dan Daftar Gambar menggunakan field Microsoft Word. Sebelum dokumen diserahkan atau diekspor menjadi PDF di Microsoft Word:

1. buka `Proposal_Tesis_USU.docx`;
2. tekan `Ctrl+A`;
3. tekan `F9`;
4. pilih pembaruan seluruh tabel jika Word meminta pilihan;
5. periksa nomor halaman dan pemenggalan tabel/persamaan setelah field diperbarui;
6. simpan kembali dokumen.

Langkah ini diperlukan agar seluruh field dinamis mencerminkan nomor halaman terbaru setelah layout final dihitung oleh Microsoft Word.

## Batas source of truth

Generator tidak mengubah substansi ilmiah BAB I–III. Ia hanya menggabungkan dan memformat naskah formal dari `docs/thesis/proposal/`. Hasil eksperimen, log pilot, nama konfigurasi internal, atau artefak implementasi tidak otomatis dimasukkan ke proposal formal.
