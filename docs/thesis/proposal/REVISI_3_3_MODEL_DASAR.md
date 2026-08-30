# Catatan Revisi Subbab 3.3 Model Dasar YOLO26n

Dokumen ini mencatat keputusan revisi hasil penelaahan Subbab 3.3 dan 3.3.1. Catatan ini belum mengganti naskah utama `BAB_III_METODOLOGI_PENELITIAN.md`; perubahan final diterapkan setelah penelaahan BAB III selesai agar struktur tidak berubah berulang kali.

## 3.3 Model Dasar YOLO26n

1. Pertahankan YOLO26n sebagai model dasar dengan bobot pralatih resmi `yolo26n.pt` dan jumlah kelas keluaran mengikuti jumlah kelas final `C` yang telah ditetapkan dari audit dataset.
2. Redaksi alasan pemilihan YOLO26n perlu dibuat lebih disiplin. Klaim seperti "relatif ringan" dan "mendukung deteksi pada beberapa skala" hanya dipertahankan sebagai alasan akademik apabila didukung sumber resmi/paper YOLO26; jika tidak, cukup nyatakan YOLO26n sebagai model dasar yang dipilih untuk eksperimen utama.
3. Pertahankan prinsip fairness inisialisasi. Sumber bobot pralatih dan prosedur inisialisasi bagian keluaran untuk jumlah kelas `C` harus dibuat setara pada kondisi yang dibandingkan. Detail seed tetap dijelaskan di Subbab 3.6 agar tidak terjadi pengulangan.
4. Pertahankan batas modifikasi arsitektur: backbone, neck, dan detection head YOLO26n tidak dimodifikasi pada eksperimen utama. Prapemrosesan juga tidak menambahkan parameter yang dilatih. Tujuannya adalah mengisolasi pengaruh perubahan representasi citra masukan.

## 3.3.1 Kondisi Eksperimen Utama dan Pembanding

Judul `Model Acuan dan Pembanding` direvisi menjadi `Kondisi Eksperimen Utama dan Pembanding` karena arsitektur model pada B0-B3 tetap YOLO26n; yang berubah adalah kondisi/pipeline prapemrosesan.

Empat kondisi utama tetap dipertahankan:

| Kode | Kondisi | Peran |
|---|---|---|
| `B0` | YOLO26n tanpa prapemrosesan tambahan | Model/kondisi acuan |
| `B1` | CLAHE + YOLO26n | Kontrol peningkatan kontras lokal konvensional |
| `B2` | `C0` + YOLO26n | Konfigurasi referensi frekuensi-angular |
| `B3` | `C*` + YOLO26n | Konfigurasi frekuensi-angular terpilih |

Keputusan tambahan:

- CLAHE tetap digunakan sebagai kontrol konvensional dengan satu konfigurasi tetap. CLAHE tidak dituning sebagai metode optimasi kedua dan nilai yang digunakan tidak boleh disebut optimum untuk citra biji kopi.
- Fungsi perbandingan utama diperjelas: `B2-B0` mengukur efek konfigurasi referensi frekuensi-angular; `B3-B2` mengukur kontribusi optimasi desain; `B3-B1` membandingkan konfigurasi terpilih dengan peningkatan kontras lokal biasa.
- `C*` tidak dipaksa berbeda dari `C0`. Apabila konfigurasi referensi `C0` menjadi kandidat terbaik berdasarkan aturan validasi yang telah ditetapkan, maka `C* = C0` merupakan hasil yang sah.
- Paragraf mengenai wavelet dipadatkan sebagai keputusan ruang lingkup penelitian, bukan sebagai klaim bahwa wavelet tidak cocok atau lebih buruk. Wavelet tidak menjadi pembanding utama karena membuka ruang keputusan tambahan (jenis wavelet, level dekomposisi, subband, ambang, rekonstruksi) di luar fokus utama.
- RT-DETRv3-R18 tetap ditempatkan sebagai analisis transfer opsional. Tujuannya adalah menguji apakah arah pengaruh `C*` muncul pada keluarga arsitektur deteksi lain, bukan untuk memilih arsitektur terbaik dan bukan untuk menentukan `C*`.

## Prinsip yang Harus Dipertahankan

Perbandingan utama harus tetap dapat dibaca sebagai:

`same YOLO26n + same data + same initialization procedure + same training settings + different input treatment`.

Dengan demikian, interpretasi utama penelitian tetap difokuskan pada pengaruh prapemrosesan citra, bukan pada perubahan arsitektur model atau perbedaan kondisi awal pelatihan.
