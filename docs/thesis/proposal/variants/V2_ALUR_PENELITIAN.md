# V2 — Diagram Alur Penelitian Multi-Dataset Publik

Status: **WORKING VARIANT — NOT FORMAL**

Diagram ini merupakan alur penelitian khusus untuk `V2_MULTI_DATASET_PUBLIK.md`. Diagram tidak digunakan oleh versi dataset primer.

![Alur penelitian multi-dataset publik](assets/alur_penelitian_multi_dataset_publik.svg){width=12.5cm}

**Gambar V2.1 Alur Penelitian Multi-Dataset Publik**

Urutan diagram:

1. candidate pool dataset publik;
2. audit provenance, lisensi, versi, anotasi, dan duplikasi;
3. pembekuan dataset independen dan split leakage-safe;
4. pemilihan dataset pengembangan $D_{dev}$;
5. pembentukan model acuan YOLO26n pada $D_{dev}$;
6. konfigurasi referensi frekuensi-angular $C_0$;
7. pengujian variasi $C_0$–$C_5$ hanya pada $D_{dev}$;
8. pemilihan dan pembekuan $C^*$;
9. konfirmasi multi-dataset menggunakan beberapa seed tanpa tuning ulang $C^*$;
10. evaluasi akhir dan analisis lintas dataset.

Asset SVG menggunakan **solid fill** dan stroke eksplisit seperti diagram arsitektur proposal. Tidak digunakan gradient, transparansi, atau fill berbasis CSS eksternal, sehingga warna lebih stabil saat dikonversi ke DOCX/PDF.
