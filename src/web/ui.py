import streamlit as st
import requests
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- 1. MODEL İÇİN UNVAN EŞLEŞTİRME ---
def map_title_to_model(selected_title):
    if "Mr (" in selected_title: return "Mr"
    if "Mrs (" in selected_title: return "Mrs"
    if "Miss (" in selected_title: return "Miss"
    if "Master (" in selected_title: return "Master"
    return "Rare"

# --- 2. SABİT VERİLER ---
GENDER_MAP = {
    "Mr (Bay)": "male", "Mrs (Evli Hanım)": "female", "Miss (Genç/Bekar Hanım)": "female",
    "Master (Genç Bey)": "male", "Rev (Rahip)": "male", "Col (Albay)": "male",
    "Major (Binbaşı)": "male", "Capt (Kaptan)": "male", "Countess (Kontes)": "female",
    "Lady (Leydi)": "female", "Sir (Sir)": "male", "Dr (Doktor)": None
}

PCLASS_OPTIONS = {
    1: "First Class — Üst Güverte (Aristokrasi)",
    2: "Second Class — Orta Güverte (Memurlar)",
    3: "Third Class — Alt Güverte (İşçiler)"
}

EMBARKED_OPTIONS = {
    "S": "Southampton, İngiltere",
    "C": "Cherbourg, Fransa",
    "Q": "Queenstown, İrlanda"
}

FARE_RANGES = {
    (1, "S"): (52.0, 150.0, 52.0), (1, "C"): (78.0, 200.0, 78.0), (1, "Q"): (90.0, 180.0, 90.0),
    (2, "S"): (13.5, 50.0, 13.5),  (2, "C"): (24.0, 60.0, 24.0),  (2, "Q"): (12.5, 45.0, 12.5),
    (3, "S"): (8.0, 30.0, 8.0),    (3, "C"): (7.9, 25.0, 7.9),    (3, "Q"): (7.75, 25.0, 7.75),
}

# --- 3. ARAYÜZ BAŞLIĞI ---
st.title("🚢 Titanic: Kader Simülasyonu")
st.markdown("15 Nisan 1912 gecesi o devasa gemide olduğunuzu hayal edin.")
st.divider()

# --- 4. DİNAMİK GİRİŞLER ---
user_name = st.text_input("Adınız ve Soyadınız", placeholder="Örn: Yasemin Karakan")

selected_display_title = st.selectbox("Size nasıl hitap edilirdi? (Unvan)", list(GENDER_MAP.keys()))

st.subheader("Yolculuk Detayları")
c_top1, c_top2 = st.columns(2)
with c_top1:
    pclass = st.selectbox("Yolcu Sınıfı", options=list(PCLASS_OPTIONS.keys()), format_func=lambda x: PCLASS_OPTIONS[x])
with c_top2:
    embarked_code = st.selectbox("Biniş Limanı", options=list(EMBARKED_OPTIONS.keys()), format_func=lambda x: EMBARKED_OPTIONS[x])

f_min, f_max, f_def = FARE_RANGES.get((pclass, embarked_code), (0.0, 250.0, 30.0))

fare = st.slider(
    "Bilet Fiyatı (£)",
    min_value=float(f_min),
    max_value=float(f_max),
    value=float(f_def),
    step=0.5
)

st.divider()

# --- 5. TAHMİN FORMU ---
with st.form("prediction_form"):
    st.subheader("Profil ve Aile Bilgileri")
    col1, col2 = st.columns(2)

    with col1:
        auto_gender = GENDER_MAP.get(selected_display_title)
        if auto_gender:
            sex = auto_gender
            st.info(f"Cinsiyet: **{sex}** (Otomatik)")
        else:
            sex = st.selectbox("Cinsiyetinizi seçin", ["male", "female"])
        age = st.slider("Yaş", 0, 100, 25)

    with col2:
        sibsp = st.number_input("Gemideki Kardeş / Eş Sayısı", 0, 10, 0)
        parch = st.number_input("Gemideki Ebeveyn / Çocuk Sayısı", 0, 10, 0)

    submit = st.form_submit_button("🔮 Kaderimi Hesapla", use_container_width=True)

# --- 6. TAHMİN MANTIĞI ---
if submit:
    title_for_model = map_title_to_model(selected_display_title)
    
    payload = {
        "Pclass": pclass,
        "Name": f"Doe, {title_for_model}. John",
        "Sex": sex,
        "Age": float(age),
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "UNKNOWN",
        "Fare": fare,
        "Cabin": None,
        "Embarked": embarked_code
    }

    try:
        api_url = os.getenv("API_URL", "http://api:8000")
        response = requests.post(f"{api_url}/predict", json=payload)
        response.raise_for_status()
        res = response.json()

        st.divider()
        greeting = f"Sayın {user_name}," if user_name else "Yolcu,"
        st.subheader(f"Simülasyon Sonucu: {greeting}")
        
        prob = res.get("survival_probability", 0) * 100
        st.progress(int(prob), text=f"Hayatta Kalma Şansı: %{prob:.1f}")

        if res.get("survived") == 1:
            st.success(f"### HAYATTASINIZ! \n\n {greeting} %{prob:.1f} olasılıkla filikalara ulaşmayı başardınız.")
            st.balloons()
        else:
            st.error(f"### KAYIP... \n\n {greeting} %{100-prob:.1f} olasılıkla o gece dondurucu sulara gömüldünüz.")

    except Exception as e:
        st.error(f"⚠️ API Bağlantı Hatası: {e}")