# §3.5 Preprocessing Frekuensi-Angular AF2 — Primary-Source Hardened

Status: **authoritative assembly replacement for §3.5 in `05_METHODOLOGY.md`**.

Use this module when assembling the formal proposal. It supersedes the older §3.5 block embedded in `05_METHODOLOGY.md` because it explicitly separates the parent-paper mechanism from repository transfer choices.

---

## 3.5 Preprocessing Frekuensi-Angular AF2

AF2 pada penelitian ini merupakan preprocessing pada ruang input yang diadaptasi dari mekanisme **patch-specific chaotic amplitude suppressor (AFAB-2)** pada LFDet yang diperkenalkan Xu et al. [FG-01]. Pemilihan istilah ini penting karena AFAB lengkap pada penelitian Xu et al. terdiri atas dua subkomponen, yaitu AFAB-1 berupa *patch-specific adaptive high-pass filter* dan AFAB-2 berupa *patch-specific chaotic amplitude suppressor*. Penelitian ini menggunakan jalur `mode=af2`, sehingga mekanisme radial high-pass AFAB-1 tidak diaktifkan. Dengan demikian, AF2 tidak boleh disamakan dengan keseluruhan AFAB milik Xu et al.

Pada LFDet, AFAB ditempatkan pada **data space/input image** sebelum proses ekstraksi fitur utama, sedangkan komponen CGFI bekerja pada feature space dan FTIF pada jalur klasifikasi fine-grained [FG-01, p. 4, §3.3, Fig. 1–2]. Posisi ini menjadi dasar penempatan AF2 sebelum YOLO26n pada penelitian ini. Namun, beberapa keputusan implementasi yang diperlukan untuk mentransfer mekanisme tersebut ke pipeline YOLO26—seperti pemrosesan RGB independen, diskretisasi sudut, padding, overlap averaging, dan residual gate eksak—merupakan keputusan repository dan tidak dinyatakan sebagai bagian identik dari metode asli.

### 3.5.1 Pembentukan patch lokal

Untuk tensor citra masukan:

\[
I\in\mathbb{R}^{B\times 3\times H\times W},
\]

citra dibagi menjadi patch lokal berukuran:

\[
m\times m,
\qquad m=32.
\]

Penggunaan DFT secara lokal pada patch dan nilai \(m=32\) berasal langsung dari §3.3.1 Xu et al. [FG-01, p. 5]. Penulis menggunakan sliding window karena respons Fourier global dianggap kurang mampu mempertahankan variasi detail lokal yang dibutuhkan pada fine-grained recognition. Paper tersebut juga menggunakan overlap yang besar untuk mengurangi diskontinuitas antarpatch dan pseudo high-frequency component pada batas patch [FG-01, p. 5].

Pada implementasi penelitian ini, overlap dibekukan menjadi:

\[
o=0.50,
\]

sehingga stride adalah:

\[
s=\operatorname{round}(m(1-o))=16.
\]

Nilai overlap 0,50 adalah konfigurasi repository, bukan klaim bahwa 0,50 merupakan nilai universal optimum. Xu et al. sendiri menunjukkan pada analisis sensitivity overlap bahwa rasio overlap berhubungan dengan trade-off akurasi dan kecepatan yang dapat berbeda antar-dataset [FG-01, p. 17, Fig. 9 dan Table 12].

Jika dimensi citra tidak tepat terhadap grid patch, implementasi repository menambahkan padding pada sisi kanan/bawah menggunakan mode `replicate`. Keputusan padding ini merupakan pilihan engineering penelitian.

### 3.5.2 Transformasi Fourier lokal

Xu et al. memperkenalkan 2D-DFT pada §3.1.1 sebagai dasar matematis pemrosesan frekuensi [FG-01, p. 3, Eq. (1)]. Secara umum, untuk sebuah patch \(P_i\), transformasi dapat ditulis:

\[
F_i(u,v)=\mathcal{F}\{P_i\}(u,v).
\]

Komponen kompleks tersebut dapat dinyatakan melalui magnitude/amplitude dan phase:

\[
A_i(u,v)=|F_i(u,v)|,
\]

\[
\phi_i(u,v)=\arg F_i(u,v).
\]

Pada repository, transformasi diimplementasikan menggunakan:

\[
F_i=\operatorname{fftshift}
\left(\operatorname{FFT2}_{\mathrm{ortho}}(P_i)\right).
\]

Normalisasi `ortho`, penggunaan `fftshift`, dan eksekusi FFT dalam `float32` ketika berada di lingkungan CUDA/AMP merupakan keputusan implementasi untuk konsistensi numerik; detail tersebut tidak diklaim sebagai formulasi identik dari Eq. (1) Xu et al.

### 3.5.3 Angular spectral density

AFAB-2 menggunakan distribusi angular density untuk merangkum intensitas amplitude berdasarkan arah frekuensi. Xu et al. mendefinisikan [FG-01, §3.3.3, Eq. (9)]:

\[
D_i^{P}(\theta)
=
\sum_r A_i^{P}(r\cos\theta,r\sin\theta),
\qquad
\theta\in[0,360^\circ).
\]

Dalam implementasi penelitian ini, domain kontinu tersebut didiskretkan menjadi:

\[
K=360
\]

bin angular. Untuk koordinat Fourier \((u,v)\), arah dihitung dengan:

\[
\theta(u,v)=
\operatorname{mod}
\left[
\operatorname{atan2}(v-v_c,u-u_c),
360^\circ
\right],
\]

kemudian dipetakan menggunakan *floor-to-bin*. Untuk channel \(c\), angular density diskret menjadi:

\[
D_i^c(k)=
\sum_{(u,v):b(u,v)=k}
A_i^c(u,v).
\]

Diskretisasi tepat 360 bin, *floor-to-bin*, serta pemrosesan setiap channel RGB secara independen adalah keputusan transfer repository. Paper induk memberi domain angular kontinu, tetapi tidak dijadikan dasar untuk mengklaim bahwa aturan diskret repository identik dengan implementasi asli penulis.

Density selanjutnya dinormalisasi menjadi distribusi probabilitas:

\[
p_i^c(k)=
\frac{D_i^c(k)}{
\sum_j D_i^c(j)+\varepsilon
}.
\]

### 3.5.4 Entropy-adaptive threshold

Xu et al. menjelaskan pada §3.3.3 bahwa entropy dari angular density digunakan untuk membentuk threshold adaptif per patch, kemudian arah dengan normalized density rendah disupresi. Implementasi repository memetakan fungsi ini ke Eq. (10)–(11) AFAB-2 dan membekukan:

\[
H_i^c
=-\sum_k
p_i^c(k)
\log\left(p_i^c(k)+\varepsilon\right),
\]

\[
\tau_i^c
=
\frac{\gamma}
{1+\exp(-H_i^c)},
\qquad
\gamma=0.10.
\]

Karena \(H_i^c\) dihitung dari patch/channel yang sedang diproses, \(\tau_i^c\) bersifat *content-adaptive* walaupun tidak berasal dari parameter trainable.

**Catatan provenance:** repository memberi anotasi eksplisit bahwa fungsi threshold mengikuti AFAB-2 Eq. (10)–(11), tetapi audit proposal saat ini belum memperoleh kembali tangkapan halaman primer yang page-perfect untuk seluruh blok Eq. (10)–(13). Oleh karena itu, nomor halaman untuk Eq. (10)–(13) tidak dicantumkan sampai recertification halaman selesai. Ini merupakan gap locator sitasi, bukan gap definisi implementasi.

### 3.5.5 Directional weighting

Density angular dinormalisasi terhadap nilai maksimum:

\[
q_i^c(k)=
\frac{D_i^c(k)}
{\max_jD_i^c(j)+\varepsilon}.
\]

Reference AF2 menerapkan hard suppression:

\[
w_i^c(k)=
\begin{cases}
0,
& q_i^c(k)\le\tau_i^c,\\
q_i^c(k),
& q_i^c(k)>\tau_i^c.
\end{cases}
\]

Bobot directional kemudian dipetakan kembali ke setiap koordinat Fourier:

\[
\widetilde F_i^c(u,v)
=
F_i^c(u,v)
\,w_i^c(b(u,v)).
\]

Secara konseptual, operasi ini mengubah magnitude/amplitude menurut distribusi angular sambil mempertahankan phase dari koefisien kompleks yang sama. Hal ini konsisten dengan deskripsi AFAB-2 Xu et al., yang menyesuaikan amplitude dan menggunakan original phase dalam rekonstruksi iDFT [FG-01, §3.3.3].

### 3.5.6 Inverse FFT dan rekonstruksi spasial

Patch yang telah dibobotkan dikembalikan ke ruang spasial:

\[
\widetilde P_i
=
\Re\left\{
\mathcal{F}^{-1}
\left(
\operatorname{ifftshift}(\widetilde F_i)
\right)
\right\}.
\]

Xu et al. menggunakan patch-wise iDFT untuk merekonstruksi respons spasial [FG-01, p. 5, §3.3.1]. Pada repository penelitian ini, patch yang saling overlap digabung menggunakan `fold`, lalu dibagi dengan `fold` terhadap tensor satu untuk memperoleh normalized overlap averaging. Prosedur reducer tersebut merupakan keputusan implementasi penelitian.

Hasilnya dinotasikan sebagai:

\[
R_{AF2}(I).
\]

### 3.5.7 Residual image enhancement

Respons spasial dinormalisasi per citra dan per channel:

\[
G(I)
=
\operatorname{MinMax}
\left(R_{AF2}(I)\right).
\]

Implementasi akhir menggunakan residual image gate:

\[
\boxed{
I'
=
I+I\odot G(I)
}
\]

sehingga:

\[
\operatorname{shape}(I')
=
\operatorname{shape}(I).
\]

Xu et al. menjelaskan recovered space sebagai gate yang mengontrol aliran informasi pada raw spatial domain [FG-01, p. 4–5, Fig. 2 dan §3.3]. Bentuk eksak `raw + raw * minmax(recovered)` adalah formulasi yang dibekukan pada repository penelitian ini.

Preservasi shape menunjukkan bahwa AF2 tidak melakukan crop, resize, translasi, atau warp koordinat bounding box. Namun, properti geometris pada preprocessing ini tidak berarti prediksi bounding box hasil training dijamin tetap identik.

### 3.5.8 Batas sumber metode dan keputusan transfer

Agar provenance metode dapat diaudit, asal setiap keputusan utama dirangkum pada Tabel 3.2.

#### Tabel 3.2 Asal keputusan desain AF2

| Elemen | Xu et al. [FG-01] | Adaptasi repository penelitian |
|---|---|---|
| patch-wise local DFT | langsung | dipertahankan |
| patch size | \(m=32\) | dipertahankan |
| overlap | paper: overlap besar; sensitivity 0.5/0.75 | dibekukan 0.50 |
| angular domain | \([0,360^\circ)\) | 360 discrete bins |
| angular density | Eq. (9) | scatter-add magnitude per bin |
| entropy-adaptive threshold | langsung pada §3.3.3 | formula dibekukan, \(\gamma=0.10\) |
| low-density directional suppression | langsung | hard threshold |
| phase | original phase dipertahankan | complex FFT coefficient retained |
| RGB processing | tidak dispesifikkan pada teks primer yang dipakai | independent RGB |
| angular discretization | tidak dispesifikkan | floor-to-bin |
| padding | tidak dispesifikkan | `replicate` |
| overlap reducer | detail eksak tidak dispesifikkan pada teks primer yang dipakai | `fold` + overlap average |
| FFT precision | tidak menjadi klaim paper | float32 pada CUDA/AMP |
| residual gate | recovered-space gating dijelaskan | `I + I * MinMax(R)` |
| learned parameters | AFAB frequency operation bukan head trainable tersendiri pada transfer ini | 0 learned parameter pada frontend |

### 3.5.9 Konfigurasi reference AF2

#### Tabel 3.3 Konfigurasi reference AF2

| Parameter | Nilai | Status provenance |
|---|---:|---|
| mode | `af2` | thesis transfer: AFAB-2-like only |
| patch size | 32 | parent + repository |
| overlap | 0,50 | repository fixed value; parent supports overlap as design variable |
| gamma | 0,10 | repository mapping to AFAB-2 threshold |
| angular bins | 360 | repository discretization |
| chunk size | 128 | engineering/memory control |
| epsilon | \(10^{-8}\) | numerical stability |
| channel processing | independent RGB | repository transfer choice |
| reconstruction | `fold` + overlap averaging | repository transfer choice |
| FFT compute | float32 under CUDA/AMP | repository engineering choice |
| learned parameter | 0 | property of this frontend implementation |

`radius_ratio=0.05` terdapat pada konfigurasi bersama AFAB karena konfigurasi yang sama juga mendukung mode `af1` dan `af12`. Parameter tersebut hanya digunakan oleh jalur radial AFAB-1 dan **tidak aktif pada `mode=af2`**. Oleh karena itu, `radius_ratio` tidak diperlakukan sebagai parameter AF2 dalam analisis penelitian ini.

### 3.5.10 Implikasi parent ablation terhadap desain optimasi

Ablation parent memberikan alasan tambahan untuk tidak melakukan module stacking secara otomatis. Pada Table 6 Xu et al., AFAB-2 secara individual meningkatkan MAR20 mAP50 dari 82,90 menjadi 84,21, sedangkan kombinasi AFAB-1+AFAB-2 menghasilkan 83,56. Pada FAIRPlane11-2.0, baseline 45,20 meningkat menjadi 45,64 dengan AFAB-2, tetapi kombinasi AFAB-1+AFAB-2 hanya mencapai 45,30 [FG-01, p. 13, §4.4.1, Table 6].

Temuan tersebut hanya menunjukkan **non-additivity pada benchmark aircraft milik parent paper**. Penelitian ini tidak mentransfer angka tersebut sebagai bukti efektivitas pada kopi. Implikasinya untuk desain eksperimen adalah lebih sempit: perubahan struktur AF2 sebaiknya diuji secara terfaktor, sehingga setiap perubahan dapat dihubungkan dengan treatment yang jelas.

**Gambar 3.3 pada dokumen final:** `RGB -> overlapping patch -> FFT -> amplitude/phase -> angular density -> entropy threshold -> directional weighting -> weighted complex spectrum -> IFFT -> overlap reconstruction -> min-max -> residual enhancement`. Diagram harus membedakan elemen yang berasal dari parent method dengan keputusan transfer repository.