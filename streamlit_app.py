from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Data Uji", layout="wide")

LABEL = {
    "jam_tidur": "Jam tidur", "tinggi": "Tinggi (cm)", "berat": "Berat (kg)",
    "bmi": "BMI", "persepsi": "Persepsi kecerdasan diri", "pd": "Kepercayaan diri (PD)",
}
BIRU = "#2f5d9e"


@st.cache_data
def load():
    d = pd.read_csv(Path(__file__).parent / "data_uji.csv")
    d["bmi"] = d["berat"] / (d["tinggi"] / 100) ** 2
    d["kat_bmi"] = pd.cut(d["bmi"], [0, 18.5, 23, 27.5, 100], right=False,
                          labels=["Kurus", "Normal", "Berat badan lebih", "Obesitas"])
    d["kat_tidur"] = pd.cut(d["jam_tidur"], [0, 3, 6, 8], labels=["1–3 jam", "4–6 jam", "7–8 jam"])
    return d


raw = load()

st.sidebar.header("Filter")
buang = st.sidebar.checkbox("Sembunyikan outlier berat (≥ 130 kg)", value=False)
d = raw[raw["berat"] < 130] if buang else raw
st.sidebar.caption(f"Data ditampilkan: n = {len(d)} dari {len(raw)}")
st.sidebar.caption("Jam tidur = jam tidur; Persepsi = penilaian diri seberapa pintar; PD = tingkat kepercayaan diri.")

st.title("Dashboard statistik deskriptif data uji")
c = st.columns(4)
c[0].metric("Responden", len(d))
c[1].metric("Rata-rata jam tidur", f"{d.jam_tidur.mean():.1f}")
c[2].metric("Rata-rata persepsi", f"{d.persepsi.mean():.2f}")
c[3].metric("Rata-rata PD", f"{d.pd.mean():.2f}")

def t_crit(df):
    """Nilai t kritis 95% dua sisi (pendekatan Cornish-Fisher, tanpa scipy)."""
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def dua_variabel(d):
    c1, c2 = st.columns(2)
    X = c1.selectbox("Variabel X (prediktor)", list(LABEL), index=4, format_func=LABEL.get, key="px")
    Y = c2.selectbox("Variabel Y (yang diprediksi)", list(LABEL), index=5, format_func=LABEL.get, key="py")
    if X == Y:
        st.info("Pilih dua variabel yang berbeda.")
        return
    n = len(d)
    x, y = d[X].to_numpy(float), d[Y].to_numpy(float)
    sxx = ((x - x.mean()) ** 2).sum()
    if n < 4 or sxx == 0:
        st.info("Data tidak cukup untuk membuat model prediksi.")
        return

    st.markdown(f"#### 1. Statistik deskriptif: {LABEL[X]} dan {LABEL[Y]}")
    rows = []
    for k in (X, Y):
        v = d[k]
        mode = v.mode().iloc[0] if v.value_counts().iloc[0] > 1 else np.nan
        rows.append({"Variabel": LABEL[k], "n": len(v), "Rata-rata": v.mean(), "Median": v.median(), "Modus": mode,
                     "Standar deviasi": v.std(), "Min": v.min(), "Q1": v.quantile(.25), "Q3": v.quantile(.75),
                     "Maks": v.max(), "Skewness": v.skew()})
    st.dataframe(pd.DataFrame(rows).set_index("Variabel").style.format("{:.2f}", na_rep="—"), width="stretch")
    a, b = st.columns(2)
    for col, k in zip((a, b), (X, Y)):
        col.altair_chart(alt.Chart(d).mark_bar(color=BIRU).encode(
            alt.X(f"{k}:Q", bin=alt.Bin(maxbins=10), title=LABEL[k]), alt.Y("count()", title="Jumlah"),
            tooltip=["count()"]).properties(height=220, title=LABEL[k]), width="stretch")

    st.markdown("#### 2. Hubungan dan model regresi linear")
    r = float(np.corrcoef(x, y)[0, 1])
    rho = d[X].corr(d[Y], method="spearman")
    b1 = r * y.std(ddof=1) / x.std(ddof=1)
    b0 = y.mean() - b1 * x.mean()
    resid = y - (b0 + b1 * x)
    dfree = n - 2
    se = float(np.sqrt((resid ** 2).sum() / dfree))
    tc = t_crit(dfree)
    se_b1 = se / np.sqrt(sxx)
    lo_b, hi_b = b1 - tc * se_b1, b1 + tc * se_b1
    m = st.columns(4)
    m[0].metric("Korelasi Pearson (r)", f"{r:.2f}")
    m[1].metric("Korelasi Spearman", f"{rho:.2f}")
    m[2].metric("R²", f"{r * r:.2f}")
    m[3].metric("Galat baku model", f"{se:.2f}")
    st.markdown(f"**Model:** {LABEL[Y]} = {b0:.2f} + {b1:.3f} × {LABEL[X]}  \n"
                f"Interval kepercayaan 95% untuk kemiringan: {lo_b:.3f} sampai {hi_b:.3f}.")

    def pred(v):
        yh = b0 + b1 * v
        return yh, tc * se * np.sqrt(1 + 1 / n + (v - x.mean()) ** 2 / sxx)

    grid = np.linspace(x.min(), x.max(), 60)
    yh, hw = pred(grid)
    band = pd.DataFrame({"x": grid, "yh": yh, "lo": yh - hw, "hi": yh + hw})
    cnt = d.groupby([X, Y]).size().reset_index(name="n")
    ex = alt.X("x:Q", title=LABEL[X], scale=alt.Scale(zero=False))
    area = alt.Chart(band).mark_area(opacity=.18, color=BIRU).encode(ex, alt.Y("lo:Q", title=LABEL[Y]), alt.Y2("hi:Q"))
    line = alt.Chart(band).mark_line(color="#c4572b", strokeWidth=2).encode(ex, alt.Y("yh:Q"))
    pts = alt.Chart(cnt).mark_circle(color=BIRU, opacity=.75).encode(
        alt.X(f"{X}:Q"), alt.Y(f"{Y}:Q"), alt.Size("n:Q", title="Jumlah responden", scale=alt.Scale(range=[60, 500])),
        tooltip=[X, Y, "n"])

    st.markdown("#### 3. Prediksi")
    integer = bool(np.all(x == np.round(x)))
    v = st.slider(f"Nilai {LABEL[X]} yang ingin diprediksi", float(x.min()), float(x.max()),
                  float(round(x.mean())) if integer else float(x.mean()), step=1.0 if integer else 0.1)
    p, h = pred(v)
    pm = st.columns(3)
    pm[0].metric(f"Prediksi {LABEL[Y]}", f"{p:.2f}")
    pm[1].metric("Batas bawah (95%)", f"{p - h:.2f}")
    pm[2].metric("Batas atas (95%)", f"{p + h:.2f}")
    if p + h > y.max() or p - h < y.min():
        st.caption(f"Catatan: rentang prediksi melewati nilai di data ({y.min():.0f}–{y.max():.0f}). Anggap nilai tertinggi/terendah di data sebagai batas sebenarnya.")
    mark = alt.Chart(pd.DataFrame({"x": [v], "y": [p]})).mark_point(shape="diamond", size=220, color="#c4572b", filled=True).encode(
        ex, alt.Y("y:Q"))
    st.altair_chart((area + pts.encode(alt.X(f"{X}:Q", title=LABEL[X]), alt.Y(f"{Y}:Q")) + line + mark).properties(height=380),
                    width="stretch")
    st.caption("Lingkaran: responden (makin besar = makin banyak yang bernilai sama). Garis oranye: prediksi model. "
               "Area biru: rentang prediksi 95% untuk satu responden baru. Belah ketupat: nilai yang kamu pilih.")

    uniq = np.sort(np.unique(x)) if len(np.unique(x)) <= 12 else np.linspace(x.min(), x.max(), 6)
    ph = [pred(u) for u in uniq]
    st.dataframe(pd.DataFrame({LABEL[X]: uniq, f"Prediksi {LABEL[Y]}": [q[0] for q in ph],
                               "Batas bawah 95%": [q[0] - q[1] for q in ph],
                               "Batas atas 95%": [q[0] + q[1] for q in ph]}).style.format("{:.2f}"),
                 width="stretch", hide_index=True)

    st.markdown("#### 4. Cek kualitas model")
    rd = pd.DataFrame({"fitted": b0 + b1 * x, "resid": resid})
    st.altair_chart((alt.Chart(rd).mark_circle(color=BIRU, opacity=.7, size=70).encode(
        alt.X("fitted:Q", title=f"Prediksi {LABEL[Y]}", scale=alt.Scale(zero=False)), alt.Y("resid:Q", title="Galat (aktual − prediksi)"))
        + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#c4572b").encode(y="z:Q")).properties(height=240),
        width="stretch")
    st.caption("Titik yang tersebar acak di sekitar garis nol menandakan model linear masuk akal. Pola melengkung atau melebar menandakan model kurang cocok.")

    arah = "naik" if b1 > 0 else "turun"
    sig = "tidak mencakup 0, jadi hubungannya cukup meyakinkan secara statistik" if lo_b > 0 or hi_b < 0 else "mencakup 0, jadi hubungannya belum bisa dipastikan"
    kuat = "kuat" if abs(r) >= .7 else "sedang" if abs(r) >= .4 else "lemah"
    st.markdown("#### 5. Interpretasi")
    st.markdown(
        f"- Setiap kenaikan 1 poin **{LABEL[X]}**, **{LABEL[Y]}** rata-rata {arah} sekitar **{abs(b1):.2f}** poin.\n"
        f"- Hubungannya **{kuat}** (r = {r:.2f}); {LABEL[X]} hanya menjelaskan **{r * r * 100:.0f}%** variasi {LABEL[Y]}, sisanya {100 - r * r * 100:.0f}% berasal dari faktor lain.\n"
        f"- Interval 95% kemiringan ({lo_b:.2f} sampai {hi_b:.2f}) {sig}.\n"
        f"- Prediksi untuk satu orang punya rentang lebar (sekitar ±{tc * se:.1f} poin), jadi pakai model ini sebagai gambaran umum, bukan angka pasti per individu.")
    st.warning("Sampel kecil (n = {}) dan variabelnya penilaian subjektif. Hubungan bukan sebab-akibat, dan prediksi di luar rentang data ({:.0f}–{:.0f}) tidak andal.".format(n, x.min(), x.max()))


t1, t2, t3, tp, t4, t5 = st.tabs(["Ringkasan", "Distribusi", "Hubungan", "2 Variabel & Prediksi", "Insight", "Data"])

with t1:
    st.subheader("Statistik deskriptif")
    rows = []
    for k, l in LABEL.items():
        s = d[k]
        mode = s.mode().iloc[0] if s.value_counts().iloc[0] > 1 else np.nan
        rows.append({"Variabel": l, "Rata-rata": s.mean(), "Median": s.median(), "Modus": mode,
                     "Standar deviasi": s.std(), "CV %": s.std() / s.mean() * 100, "Min": s.min(),
                     "Q1": s.quantile(.25), "Q3": s.quantile(.75), "Maks": s.max(), "Skewness": s.skew()})
    st.dataframe(pd.DataFrame(rows).set_index("Variabel").style.format("{:.2f}", na_rep="—"), width="stretch")
    st.caption("Skewness mendekati 0 = simetris; jauh di atas 1 = ada nilai ekstrem besar. CV = standar deviasi ÷ rata-rata.")
    a, b = st.columns(2)
    kb = d["kat_bmi"].value_counts().reindex(["Kurus", "Normal", "Berat badan lebih", "Obesitas"]).reset_index()
    kb.columns = ["Kategori", "Jumlah"]
    a.subheader("Kategori BMI")
    a.altair_chart(alt.Chart(kb).mark_bar(color=BIRU).encode(
        alt.X("Kategori:N", sort=None, title=None), alt.Y("Jumlah:Q"), tooltip=["Kategori", "Jumlah"]
    ).properties(height=260), width="stretch")
    a.caption("Patokan Asia-Pasifik: <18,5 kurus; 18,5–22,9 normal; 23–27,4 berat badan lebih; ≥27,5 obesitas.")
    b.subheader("Jam tidur")
    jt = d["jam_tidur"].value_counts().sort_index().reset_index()
    jt.columns = ["Jam", "Jumlah"]
    b.altair_chart(alt.Chart(jt).mark_bar(color=BIRU).encode(
        alt.X("Jam:O", title="Jam tidur"), alt.Y("Jumlah:Q"), tooltip=["Jam", "Jumlah"]
    ).properties(height=260), width="stretch")

with t2:
    st.subheader("Distribusi")
    k = st.selectbox("Variabel", list(LABEL), index=4, format_func=LABEL.get)
    hist = alt.Chart(d).mark_bar(color=BIRU).encode(
        alt.X(f"{k}:Q", bin=alt.Bin(maxbins=10), title=LABEL[k]), alt.Y("count()", title="Jumlah"),
        tooltip=["count()"]).properties(height=280)
    box = alt.Chart(d).mark_boxplot(extent=1.5, color=BIRU).encode(
        alt.X(f"{k}:Q", title=LABEL[k], scale=alt.Scale(zero=False))).properties(height=90)
    st.altair_chart(hist, width="stretch")
    st.altair_chart(box, width="stretch")
    st.caption("Kotak = Q1–Q3, garis dalam kotak = median, titik di luar kumis = outlier (aturan 1,5 × IQR).")

with t3:
    st.subheader("Korelasi antar variabel")
    cr = (d[list(LABEL)].rename(columns=LABEL).corr().rename_axis("a").reset_index()
          .melt("a", var_name="b", value_name="r"))
    base = alt.Chart(cr).encode(alt.X("b:N", title=None), alt.Y("a:N", title=None))
    heat = base.mark_rect().encode(alt.Color("r:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])))
    txt = base.mark_text(fontSize=12).encode(
        alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.55", alt.value("white"), alt.value("black")))
    st.altair_chart((heat + txt).properties(height=380), width="stretch")
    st.caption("Pearson r: biru = positif, oranye = negatif. Dengan n sekitar 30, hanya |r| di atas ±0,36 yang cukup layak diperhatikan.")

    st.subheader("Diagram pencar")
    p, q = st.columns(2)
    kx = p.selectbox("Sumbu X", list(LABEL), index=4, format_func=LABEL.get)
    ky = q.selectbox("Sumbu Y", list(LABEL), index=5, format_func=LABEL.get)
    ch = alt.Chart(d)
    pts = ch.mark_circle(size=80, opacity=.7, color=BIRU).encode(
        alt.X(f"{kx}:Q", title=LABEL[kx], scale=alt.Scale(zero=False)),
        alt.Y(f"{ky}:Q", title=LABEL[ky], scale=alt.Scale(zero=False)), tooltip=[kx, ky])
    if kx != ky:
        pts = pts + ch.transform_regression(kx, ky).mark_line(color="#c4572b").encode(
            alt.X(f"{kx}:Q"), alt.Y(f"{ky}:Q"))
        r = d[kx].corr(d[ky])
        m = st.columns(2)
        m[0].metric("r (korelasi)", f"{r:.2f}")
        m[1].metric("R²", f"{r * r:.2f}")
    st.altair_chart(pts.properties(height=360), width="stretch")

    st.subheader("Rata-rata menurut kelompok jam tidur")
    g = d.groupby("kat_tidur", observed=True).agg(n=("pd", "size"), persepsi=("persepsi", "mean"), pd=("pd", "mean")).round(2)
    g.columns = ["Jumlah responden", "Rata-rata persepsi", "Rata-rata PD"]
    st.dataframe(g, width="stretch")

with tp:
    st.subheader("Analisis dua variabel dan prediksi")
    dua_variabel(d)

with t4:
    st.subheader("Insight")
    st.caption("Berdasarkan seluruh data (n = 30), tidak terpengaruh filter di sidebar.")
    st.markdown("""
1. **Kurang tidur cukup banyak.** Rata-rata tidur 5,2 jam. Hanya 11 dari 30 orang (37%) yang tidur 7 jam atau lebih, dan 8 orang (27%) tidur 4 jam atau kurang. Nilai 1–2 jam (6 orang) perlu dicek apakah benar.
2. **Penilaian kecerdasan diri tergolong tinggi.** Nilai 9 paling sering dipilih (7 orang); 10 orang (33%) memberi 8 ke atas.
3. **Kepercayaan diri juga condong tinggi.** Nilai 9 dipilih 11 orang (37%); setengah responden bernilai 8 ke atas. Rata-rata PD (7,0) sedikit di atas persepsi (6,57).
4. **Merasa pintar berkaitan dengan percaya diri** (r = 0,49, sedang). Ini hubungan terkuat di data; 13 dari 30 orang (43%) bernilai sama persis di keduanya.
5. **Tidur paling sedikit, tetapi merasa paling pintar dan percaya diri.** Kelompok 1–3 jam: persepsi 7,57 dan PD 7,71, lebih tinggi dari kelompok lain. Korelasinya lemah (−0,27 dan −0,20) dan kelompok ini hanya 7 orang, jadi hanya indikasi awal. Jangan dibaca bahwa kurang tidur membuat orang lebih pintar; persepsi hanyalah penilaian diri.
6. **Berat badan dipengaruhi dua outlier** (130 kg dan 140 kg). Rata-rata 63,2 kg, median 58 kg; tanpa keduanya rata-rata 58,1 kg. Gunakan median, atau centang filter di sidebar.
7. **Tinggi, berat, dan BMI tidak terkait dengan persepsi maupun PD** (r sekitar 0 sampai 0,08).
""")
    st.warning(
        "**Catatan kualitas data**\n\n"
        "- Seluruh data masuk dalam sekitar 4 menit, kemungkinan ada yang diisi asal.\n"
        "- Satu baris identik muncul dua kali, dan pasangan tinggi 170 cm dengan berat 60 kg muncul 4 kali.\n"
        "- Sampel hanya 30 orang, jadi hasil tidak bisa digeneralisasi. Persepsi dan PD adalah penilaian subjektif."
    )

with t5:
    st.subheader("Data mentah")
    st.dataframe(d.rename(columns=LABEL | {"timestamp": "Waktu", "kat_bmi": "Kategori BMI", "kat_tidur": "Kelompok tidur"})
                 .style.format({"BMI": "{:.1f}"}), width="stretch", hide_index=True)
