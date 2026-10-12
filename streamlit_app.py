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

t1, t2, t3, t4, t5 = st.tabs(["Ringkasan", "Distribusi", "Hubungan", "Insight", "Data"])

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
