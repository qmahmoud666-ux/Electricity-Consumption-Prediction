import streamlit as st
import joblib
import pandas as pd

st.title("⚡ تطبيق توقع استهلاك الطاقة الكهربائية")
st.write("أدخلي بيانات الجهاز والبيئة المحيطة لحساب الاستهلاك المتوقع (kWh):")

@st.cache_resource
def load_model():
    return joblib.load('best_tuned_model.pkl')

model = load_model()

# إنشاء واجهة المدخلات
st.header("بيانات الاستخدام")

col1, col2 = st.columns(2)

with col1:
    appliance_type = st.selectbox(
        "نوع الجهاز", 
        ['washing machine', 'refrigerator', 'TV', 'heater', 'Air Conditioner', 'Microwave', 'AC']
    )
    appliance_power_watts = st.number_input("قدرة الجهاز بالواط", value=500.0, step=50.0)
    usage_hours = st.number_input("ساعات الاستخدام", value=5.0, step=0.5)
    temperature_c = st.number_input("درجة الحرارة", value=25.0, step=1.0)
    room_size_m2 = st.number_input("مساحة الغرفة m²", value=20.0, step=1.0)

with col2:
    number_of_people = st.number_input("عدد الأشخاص", value=2, step=1)
    time_of_day = st.selectbox("وقت اليوم", ['Morning', 'Afternoon', 'Evening', 'Night'])
    day_type = st.selectbox("نوع اليوم", ['Weekday', 'Weekend'])
    previous_consumption_kwh = st.number_input("الاستهلاك السابق (kWh)", value=5.0, step=0.1)

if st.button("حساب الاستهلاك المتوقع"):
    # 1. إعداد جدول المدخلات الخام
    raw_df = pd.DataFrame([{
        'appliance_type': appliance_type,
        'appliance_power_watts': appliance_power_watts,
        'usage_hours': usage_hours,
        'temperature_c': temperature_c,
        'room_size_m2': room_size_m2,
        'number_of_people': number_of_people,
        'time_of_day': time_of_day,
        'day_type': day_type,
        'previous_consumption_kwh': previous_consumption_kwh
    }])

    # 2. تحويل النصوص لأعمدة Encoding
    encoded_df = pd.get_dummies(raw_df)

    # 3. محاذاة الأعمدة لتطابق الأعمدة التي تدرب عليها الموديل تماماً
    try:
        model_features = model.feature_names_in_
        encoded_df = encoded_df.reindex(columns=model_features, fill_value=0)
        
        # 4. التوقع
        prediction = model.predict(encoded_df)
        st.success(f"💡 الاستهلاك المتوقع: {prediction[0]:,.2f} kWh")
    except Exception as e:
        st.error(f"حدث خطأ أثناء التوقع: {e}")