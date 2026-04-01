import streamlit as st
import pandas as pd
import pickle
import os

# 1. إعدادات النظام الأساسية
st.set_page_config(page_title="نظام تحليل مخاطر المشاريع المتقدم", page_icon="🛡️", layout="wide")

# 2. محرك تحميل الذكاء الاصطناعي
@st.cache_resource
def load_my_model():
    path = "ML_Model_2/xgboost_risk_model.pkl"
    if not os.path.exists(path):
        path = "xgboost_risk_model.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# تحميل الموديل
try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    st.error(f"فشل تحميل الموديل: {e}")
    model_loaded = False

# 3. الواجهة الرسومية المحترفة
st.title("🛡️ نظام التنبؤ المتقدم بمخاطر المشاريع")
st.markdown("تحليل تقني يعتمد على خوارزميات التعلم الآلي لتقييم كفاءة ومخاطر المشاريع الإدارية.")

# إنشاء تبويبات (Tabs) لتنظيم الواجهة
tab1, tab2 = st.tabs(["📝 إدخال مشروع مفرد", "📊 تحليل ملفات ضخمة (Excel/CSV)"])

# --- التبويب الأول: إدخال يدوي ---
with tab1:
    st.header("إدخال بيانات المشروع")
    col1, col2 = st.columns(2)
    with col1:
        total_tasks = st.number_input("إجمالي المهام المخطط لها", min_value=1, value=100, key="single_total")
        completed_tasks = st.number_input("المهام المكتملة", min_value=0, value=50, key="single_done")
        delayed_tasks = st.number_input("المهام المتأخرة", min_value=0, value=10, key="single_delay")
    with col2:
        budget_pct = st.slider("نسبة استهلاك الميزانية (%)", 0, 200, 70, key="single_budget")
        experience = st.selectbox("مستوى خبرة الفريق", [1, 2, 3], key="single_exp")
    
    if st.button("تحليل المشروع الفردي 🚀"):
        if model_loaded:
            completion_pct = (completed_tasks / total_tasks) * 100
            input_data = pd.DataFrame([{
                "Total_Tasks": total_tasks, "Completed_Tasks": completed_tasks,
                "Delayed_Tasks": delayed_tasks, "Budget_Spent_Pct": budget_pct,
                "Team_Experience": experience, "Completion_Pct": completion_pct
            }])
            prediction = model.predict(input_data)[0]
            st.divider()
            if prediction == 1:
                st.error("### النتيجة التقنية: المشروع في حالة خطر (At Risk)")
            else:
                st.success("### النتيجة التقنية: وضع المشروع سليم (Healthy)")
        else:
            st.warning("المحرك الذكي غير متوفر.")

# --- التبويب الثاني: تحليل الملفات الضخمة ---
with tab2:
    st.header("رفع ومعالجة البيانات الضخمة")
    st.info("ملاحظة: تأكد أن ملف Excel أو CSV يحتوي على الأعمدة المطلوبة.")
    
    uploaded_file = st.file_uploader("اختر ملف Excel أو CSV", type=["csv", "xlsx"])
    
    if uploaded_file is not None and model_loaded:
        # قراءة الملف بذكاء حسب النوع
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write(f"تم تحميل ملف يحتوي على {len(df)} صف.")
            
            # التحقق من وجود الأعمدة المطلوبة
            required_cols = ["Total_Tasks", "Completed_Tasks", "Delayed_Tasks", "Budget_Spent_Pct", "Team_Experience"]
            if all(col in df.columns for col in required_cols):
                
                # معالجة البيانات ضخمة الحجم آلياً
                df['Completion_Pct'] = (df['Completed_Tasks'] / df['Total_Tasks']) * 100
                
                # تنفيذ التنبؤ على كامل الملف دفعة واحدة
                features = df[required_cols + ["Completion_Pct"]]
                df['Risk_Prediction'] = model.predict(features)
                
                # تحويل النتائج لنص مفهوم
                df['Status'] = df['Risk_Prediction'].map({1: "At Risk (خطر)", 0: "Healthy (سليم)"})
                
                st.divider()
                st.subheader("نتائج التحليل:")
                st.dataframe(df.style.map(lambda x: 'color: red' if 'خطر' in str(x) else 'color: green', subset=['Status']))
                
                # خيار تحميل النتائج
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("تحميل ملف النتائج (CSV) 📥", data=csv_data, file_name="Project_Risk_Analysis.csv")
                
            else:
                st.error(f"الملف لا يحتوي على كل الأعمدة المطلوبة: {required_cols}")
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
