# Generator PPT Seminar Proposal

Generator ini membuat PPT seminar proposal dari `docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md` dengan **PPT referensi pengguna sebagai template visual**.

## Prinsip

- Tidak membuat branch baru.
- Naskah formal BAB I–III tetap menjadi sumber ilmiah utama.
- `PPT_SEMINAR_PROPOSAL.md` hanya berisi teks ringkas untuk presentasi.
- Template PPTX menjadi sumber format visual: ukuran slide, tema, warna, latar, dekorasi, dan komposisi dasar.
- Generator memakai ulang slide-slide template sebagai archetype layout dan mengganti teksnya.

## Persiapan

```bash
pip install python-pptx
```

Simpan PPT referensi, misalnya:

```text
assets/ppt/ppt_seminar_proposal_template.pptx
```

File template biner tidak disimpan otomatis oleh generator. Gunakan PPT referensi yang diberikan pengguna sebagai `--template`.

## Generate

```bash
python tools/thesis_pptx/generate_seminar_proposal.py \
  --template assets/ppt/ppt_seminar_proposal_template.pptx \
  --source docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md \
  --output build/seminar_proposal_kopi.pptx
```

## Edit isi

Untuk revisi teks presentasi, edit:

```text
docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md
```

kemudian jalankan generator kembali. Jangan mengubah BAB I–III hanya untuk menyesuaikan layout slide.

## Catatan

Generator sengaja mempertahankan teks slide agar ringkas. Rumusan Masalah dan Tujuan Penelitian menggunakan bentuk naratif, sedangkan Batasan Masalah dan rincian eksperimen dapat menggunakan bullet/tabel sesuai kebutuhan presentasi.
