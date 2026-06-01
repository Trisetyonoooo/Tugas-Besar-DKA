import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="Fuzzy Attrition Predictor v3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= FUZZY LOGIC CLASSES (FROM SCRATCH) =============

class TriangularMembership:
    """Fungsi Keanggotaan Segitiga - From Scratch Implementation"""
    def __init__(self, a, b, c, name=""):
        self.a = a  # Lower bound
        self.b = b  # Peak
        self.c = c  # Upper bound
        self.name = name

    def compute(self, x):
        """Hitung derajat keanggotaan untuk nilai x"""
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
        else:
            return (self.c - x) / (self.c - self.b) if self.c != self.b else 0.0


class FuzzyVariable:
    """Variabel Fuzzy dengan multiple himpunan fuzzy"""
    def __init__(self, name, domain_min, domain_max):
        self.name = name
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.sets = {}

    def add_triangular_set(self, set_name, a, b, c):
        """Tambah himpunan fuzzy segitiga"""
        self.sets[set_name] = TriangularMembership(a, b, c, set_name)

    def fuzzify(self, value):
        """Fuzzifikasi: ubah nilai crisp menjadi fuzzy"""
        result = {}
        for set_name, membership_func in self.sets.items():
            result[set_name] = membership_func.compute(value)
        return result

    def plot_membership(self, ax=None):
        """Visualisasi fungsi keanggotaan"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        x = np.linspace(self.domain_min, self.domain_max, 1000)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.sets)))

        for (set_name, mf), color in zip(self.sets.items(), colors):
            y = [mf.compute(val) for val in x]
            ax.plot(x, y, label=set_name, linewidth=2.5, color=color)
            ax.fill_between(x, y, alpha=0.3, color=color)

        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Membership Degree', fontsize=10)
        ax.set_title(f'{self.name} - Fungsi Keanggotaan', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        return ax


class FuzzyRule:
    """Aturan Fuzzy IF-THEN"""
    def __init__(self, conditions, conclusion, rule_id=""):
        self.conditions = conditions  # List of (var_name, fuzzy_set_name)
        self.conclusion = conclusion  # (output_var_name, fuzzy_set_name, crisp_value)
        self.rule_id = rule_id

    def evaluate(self, fuzzified_inputs):
        """Evaluasi rule dengan operator MIN (AND)"""
        strengths = []
        for var_name, fuzzy_set in self.conditions:
            if var_name in fuzzified_inputs:
                strength = fuzzified_inputs[var_name].get(fuzzy_set, 0.0)
                strengths.append(strength)
        
        if strengths:
            return min(strengths)
        return 0.0


class FuzzyMamdani:
    """Fuzzy Inference System - Metode Mamdani"""
    def __init__(self, output_variable):
        self.input_variables = {}
        self.output_variable = output_variable
        self.rules = []

    def add_input_variable(self, variable):
        self.input_variables[variable.name] = variable

    def add_rule(self, rule):
        self.rules.append(rule)

    def fuzzify_inputs(self, input_values):
        """Fuzzifikasi: ubah nilai crisp menjadi fuzzy"""
        fuzzified = {}
        for var_name, var_obj in self.input_variables.items():
            if var_name in input_values:
                fuzzified[var_name] = var_obj.fuzzify(input_values[var_name])
        return fuzzified

    def inference(self, fuzzified_inputs):
        """Inferensi: Hitung firing strength untuk setiap rule"""
        activated_rules = []
        for rule in self.rules:
            firing_strength = rule.evaluate(fuzzified_inputs)
            if firing_strength > 0.001:
                activated_rules.append({
                    'rule_id': rule.rule_id,
                    'firing_strength': firing_strength,
                    'conclusion': rule.conclusion
                })
        return activated_rules

    def defuzzify_centroid(self, activated_rules):
        """Defuzzifikasi dengan metode Centroid (Center of Gravity)"""
        if not activated_rules:
            return 0.0

        numerator = 0.0
        denominator = 0.0
        z_range = np.linspace(self.output_variable.domain_min,
                             self.output_variable.domain_max, 500)

        for z in z_range:
            max_membership = 0.0
            for activated_rule in activated_rules:
                firing_strength = activated_rule['firing_strength']
                output_set_name = activated_rule['conclusion'][1]
                membership_func = self.output_variable.sets[output_set_name]
                implication_result = min(membership_func.compute(z), firing_strength)
                max_membership = max(max_membership, implication_result)

            numerator += z * max_membership
            denominator += max_membership

        if denominator > 0:
            return numerator / denominator
        return 0.0

    def predict(self, input_values):
        """Proses lengkap: Fuzzifikasi -> Inferensi -> Defuzzifikasi"""
        fuzzified = self.fuzzify_inputs(input_values)
        activated_rules = self.inference(fuzzified)
        output = self.defuzzify_centroid(activated_rules)
        return output, fuzzified, activated_rules


class FuzzySugeno:
    """Fuzzy Inference System - Metode Sugeno (Zero-Order)"""
    def __init__(self):
        self.input_variables = {}
        self.rules = []

    def add_input_variable(self, variable):
        self.input_variables[variable.name] = variable

    def add_rule(self, rule):
        self.rules.append(rule)

    def fuzzify_inputs(self, input_values):
        """Fuzzifikasi: ubah nilai crisp menjadi fuzzy"""
        fuzzified = {}
        for var_name, var_obj in self.input_variables.items():
            if var_name in input_values:
                fuzzified[var_name] = var_obj.fuzzify(input_values[var_name])
        return fuzzified

    def inference(self, fuzzified_inputs):
        """Inferensi untuk Sugeno"""
        activated_rules = []
        for rule in self.rules:
            firing_strength = rule.evaluate(fuzzified_inputs)
            if firing_strength > 0.001:
                activated_rules.append({
                    'rule_id': rule.rule_id,
                    'firing_strength': firing_strength,
                    'conclusion': rule.conclusion
                })
        return activated_rules

    def weighted_average_defuzzify(self, activated_rules):
        """Defuzzifikasi dengan Weighted Average"""
        if not activated_rules:
            return 0.0

        numerator = 0.0
        denominator = 0.0
        for activated_rule in activated_rules:
            firing_strength = activated_rule['firing_strength']
            crisp_value = activated_rule['conclusion'][2]
            numerator += firing_strength * crisp_value
            denominator += firing_strength

        if denominator > 0:
            return numerator / denominator
        return 0.0

    def predict(self, input_values):
        """Proses lengkap Sugeno"""
        fuzzified = self.fuzzify_inputs(input_values)
        activated_rules = self.inference(fuzzified)
        output = self.weighted_average_defuzzify(activated_rules)
        return output, fuzzified, activated_rules


# ============= INITIALIZE FUZZY SYSTEMS =============

@st.cache_resource
def initialize_fuzzy_systems():
    """Initialize Fuzzy Mamdani dan Sugeno systems dengan 17 rules"""
    
    # ===== LINGUISTIC VARIABLES SETUP =====
    
    # Input Variables
    job_satisfaction = FuzzyVariable('JobSatisfaction', 1, 10)
    job_satisfaction.add_triangular_set('Rendah', 0, 1, 4)
    job_satisfaction.add_triangular_set('Sedang', 3, 5.5, 8)
    job_satisfaction.add_triangular_set('Tinggi', 6, 10, 11)

    work_life_balance = FuzzyVariable('WorkLifeBalance', 1, 10)
    work_life_balance.add_triangular_set('Rendah', 0, 1, 4)
    work_life_balance.add_triangular_set('Sedang', 3, 5.5, 8)
    work_life_balance.add_triangular_set('Tinggi', 6, 10, 11)

    overtime = FuzzyVariable('Overtime', 0, 10)
    overtime.add_triangular_set('Rendah', -1, 0, 3)
    overtime.add_triangular_set('Sedang', 2, 5, 8)
    overtime.add_triangular_set('Tinggi', 7, 10, 11)

    performance_rating = FuzzyVariable('PerformanceRating', 1, 5)
    performance_rating.add_triangular_set('Buruk', 0, 1, 2.5)
    performance_rating.add_triangular_set('Biasa', 1.5, 3, 4.5)
    performance_rating.add_triangular_set('Bagus', 3.5, 5, 6)

    hours_worked = FuzzyVariable('AvgHoursPerWeek', 20, 80)
    hours_worked.add_triangular_set('Normal', 15, 30, 45)
    hours_worked.add_triangular_set('Tinggi', 40, 55, 70)
    hours_worked.add_triangular_set('SangatTinggi', 65, 80, 90)

    # Output Variable
    attrition_risk = FuzzyVariable('AttritionRisk', 0, 100)
    attrition_risk.add_triangular_set('SangatRendah', -5, 0, 20)
    attrition_risk.add_triangular_set('Rendah', 15, 30, 45)
    attrition_risk.add_triangular_set('Sedang', 40, 50, 60)
    attrition_risk.add_triangular_set('Tinggi', 55, 70, 85)
    attrition_risk.add_triangular_set('SangatTinggi', 80, 100, 105)

    # ===== DEFINE 17 FUZZY RULES =====
    
    rules = [
        # High Risk Rules (R1-R5)
        FuzzyRule([('JobSatisfaction', 'Rendah'), ('WorkLifeBalance', 'Rendah')],
                  ('AttritionRisk', 'SangatTinggi', 90), "R1"),
        FuzzyRule([('JobSatisfaction', 'Rendah'), ('Overtime', 'Tinggi')],
                  ('AttritionRisk', 'SangatTinggi', 85), "R2"),
        FuzzyRule([('WorkLifeBalance', 'Rendah'), ('Overtime', 'Tinggi')],
                  ('AttritionRisk', 'Tinggi', 75), "R3"),
        FuzzyRule([('PerformanceRating', 'Buruk'), ('JobSatisfaction', 'Rendah')],
                  ('AttritionRisk', 'Tinggi', 70), "R4"),
        FuzzyRule([('Overtime', 'Tinggi'), ('AvgHoursPerWeek', 'SangatTinggi')],
                  ('AttritionRisk', 'SangatTinggi', 88), "R5"),

        # Medium Risk Rules (R6-R9)
        FuzzyRule([('JobSatisfaction', 'Sedang'), ('WorkLifeBalance', 'Sedang')],
                  ('AttritionRisk', 'Sedang', 50), "R6"),
        FuzzyRule([('Overtime', 'Sedang'), ('AvgHoursPerWeek', 'Tinggi')],
                  ('AttritionRisk', 'Sedang', 55), "R7"),
        FuzzyRule([('JobSatisfaction', 'Sedang'), ('Overtime', 'Sedang')],
                  ('AttritionRisk', 'Sedang', 52), "R8"),
        FuzzyRule([('PerformanceRating', 'Biasa'), ('WorkLifeBalance', 'Sedang')],
                  ('AttritionRisk', 'Sedang', 48), "R9"),

        # Low Risk Rules (R10-R13)
        FuzzyRule([('JobSatisfaction', 'Tinggi'), ('WorkLifeBalance', 'Tinggi')],
                  ('AttritionRisk', 'SangatRendah', 10), "R10"),
        FuzzyRule([('JobSatisfaction', 'Tinggi'), ('PerformanceRating', 'Bagus')],
                  ('AttritionRisk', 'Rendah', 25), "R11"),
        FuzzyRule([('WorkLifeBalance', 'Tinggi'), ('Overtime', 'Rendah')],
                  ('AttritionRisk', 'Rendah', 30), "R12"),
        FuzzyRule([('PerformanceRating', 'Bagus'), ('Overtime', 'Rendah')],
                  ('AttritionRisk', 'SangatRendah', 15), "R13"),

        # Mixed Rules (R14-R17)
        FuzzyRule([('JobSatisfaction', 'Rendah'), ('PerformanceRating', 'Bagus')],
                  ('AttritionRisk', 'Sedang', 45), "R14"),
        FuzzyRule([('Overtime', 'Rendah'), ('AvgHoursPerWeek', 'Normal')],
                  ('AttritionRisk', 'Rendah', 35), "R15"),
        FuzzyRule([('WorkLifeBalance', 'Rendah'), ('PerformanceRating', 'Buruk')],
                  ('AttritionRisk', 'Tinggi', 72), "R16"),
        FuzzyRule([('JobSatisfaction', 'Tinggi'), ('Overtime', 'Tinggi')],
                  ('AttritionRisk', 'Sedang', 58), "R17")
    ]

    # ===== INITIALIZE MAMDANI SYSTEM =====
    
    mamdani = FuzzyMamdani(attrition_risk)
    mamdani.add_input_variable(job_satisfaction)
    mamdani.add_input_variable(work_life_balance)
    mamdani.add_input_variable(overtime)
    mamdani.add_input_variable(performance_rating)
    mamdani.add_input_variable(hours_worked)
    
    for rule in rules:
        mamdani.add_rule(rule)

    # ===== INITIALIZE SUGENO SYSTEM =====
    
    sugeno = FuzzySugeno()
    sugeno.add_input_variable(job_satisfaction)
    sugeno.add_input_variable(work_life_balance)
    sugeno.add_input_variable(overtime)
    sugeno.add_input_variable(performance_rating)
    sugeno.add_input_variable(hours_worked)
    
    for rule in rules:
        sugeno.add_rule(rule)

    return mamdani, sugeno, {
        'job_satisfaction': job_satisfaction,
        'work_life_balance': work_life_balance,
        'overtime': overtime,
        'performance_rating': performance_rating,
        'hours_worked': hours_worked,
        'attrition_risk': attrition_risk
    }


# ============= HELPER FUNCTIONS =============

def get_risk_level(score):
    """Klasifikasi risk level berdasarkan score"""
    if score >= 80:
        return "🔴 SANGAT TINGGI", "very-high", "#ff4757"
    elif score >= 65:
        return "🟠 TINGGI", "high", "#ffa502"
    elif score >= 50:
        return "🟡 SEDANG", "medium", "#ffd93d"
    elif score >= 35:
        return "🟢 RENDAH", "low", "#6bcf7f"
    else:
        return "🟢 SANGAT RENDAH", "very-low", "#1dd1a1"


def get_recommendations(score, js, wlb, ot, pr, ah):
    """Rekomendasi berdasarkan risk score dan faktor-faktor utama"""
    recs = []
    
    if score >= 75:
        recs.append("🚨 URGENT: Perlukan intervensi segera!")
        if js < 4:
            recs.append("💡 Identifikasi masalah kepuasan kerja karyawan")
        if wlb < 4:
            recs.append("⚖️ Tingkatkan work-life balance (kurangi beban kerja)")
        if ot > 7:
            recs.append("⏰ Kurangi overtime dan beban kerja")
        recs.append("💬 Lakukan one-on-one meeting segera")
    elif score >= 60:
        recs.append("⚠️ Monitoring ketat diperlukan")
        if js < 5:
            recs.append("💡 Tawarkan job enrichment opportunities")
        if ot > 6:
            recs.append("⏰ Monitor dan manage overtime")
        recs.append("🎓 Tawarkan development programs")
    elif score >= 45:
        recs.append("📋 Monitoring rutin diperlukan")
        recs.append("✅ Pertahankan engagement level ini")
        recs.append("🏢 Lanjutkan team engagement activities")
    else:
        recs.append("✅ Karyawan dalam kondisi stabil")
        recs.append("🎉 Pertahankan engagement level ini")
        recs.append("🌟 Gunakan sebagai best practice reference")
    
    return recs


def plot_fuzzified_inputs(fuzzified_dict, variables):
    """Plot hasil fuzzifikasi untuk semua input variables"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Hasil Fuzzifikasi - Derajat Keanggotaan Input Variables', 
                 fontsize=12, fontweight='bold')
    axes = axes.flatten()
    
    var_names = ['JobSatisfaction', 'WorkLifeBalance', 'Overtime', 
                 'PerformanceRating', 'AvgHoursPerWeek']
    
    for idx, var_name in enumerate(var_names):
        ax = axes[idx]
        if var_name in fuzzified_dict:
            fuzzy_values = fuzzified_dict[var_name]
            sets = list(fuzzy_values.keys())
            values = list(fuzzy_values.values())
            
            colors = ['#ff6b6b' if v > 0.5 else '#4ecdc4' if v > 0.1 else '#dfe6e9' 
                     for v in values]
            bars = ax.barh(sets, values, color=colors, edgecolor='black', linewidth=1.5)
            
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
            
            ax.set_xlabel('Derajat Keanggotaan', fontsize=9)
            ax.set_title(var_name, fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.2)
            ax.grid(axis='x', alpha=0.3)
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_activated_rules(activated_rules):
    """Plot activated rules dengan firing strength"""
    if not activated_rules:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rule_ids = [r['rule_id'] for r in activated_rules]
    firing_strengths = [r['firing_strength'] for r in activated_rules]
    conclusions = [r['conclusion'][1] for r in activated_rules]
    
    # Color mapping
    colors_map = {
        'SangatRendah': '#1dd1a1',
        'Rendah': '#6bcf7f',
        'Sedang': '#ffd93d',
        'Tinggi': '#ffa502',
        'SangatTinggi': '#ff4757'
    }
    colors = [colors_map.get(c, '#999') for c in conclusions]
    
    bars = ax.barh(rule_ids, firing_strengths, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, fs, conc in zip(bars, firing_strengths, conclusions):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{fs:.3f} → {conc}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Firing Strength (Derajat Aktivasi)', fontsize=10)
    ax.set_title('Activated Fuzzy Rules (Aturan yang Diaktifkan)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_output_membership(output_var, mamdani_score, sugeno_score, activated_rules):
    """Visualisasi output membership function dengan hasil prediksi"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_range = np.linspace(output_var.domain_min, output_var.domain_max, 1000)
    
    # Plot semua membership functions
    colors_map = {
        'SangatRendah': '#1dd1a1',
        'Rendah': '#6bcf7f',
        'Sedang': '#ffd93d',
        'Tinggi': '#ffa502',
        'SangatTinggi': '#ff4757'
    }
    
    for set_name, mf in output_var.sets.items():
        y_vals = [mf.compute(x) for x in x_range]
        color = colors_map.get(set_name, '#999')
        ax.plot(x_range, y_vals, label=set_name, linewidth=2.5, color=color)
        ax.fill_between(x_range, y_vals, alpha=0.2, color=color)
    
    # Plot vertical lines untuk Mamdani dan Sugeno results
    ax.axvline(mamdani_score, color='#667eea', linestyle='--', linewidth=3, 
              label=f'Mamdani: {mamdani_score:.1f}', alpha=0.8)
    ax.axvline(sugeno_score, color='#f5576c', linestyle='--', linewidth=3, 
              label=f'Sugeno: {sugeno_score:.1f}', alpha=0.8)
    
    # Shade activated membership area
    if activated_rules:
        max_membership = np.zeros_like(x_range)
        for i, z in enumerate(x_range):
            max_mu = 0.0
            for rule in activated_rules:
                fs = rule['firing_strength']
                output_set_name = rule['conclusion'][1]
                mf = output_var.sets[output_set_name]
                implication = min(mf.compute(z), fs)
                max_mu = max(max_mu, implication)
            max_membership[i] = max_mu
        
        ax.fill_between(x_range, max_membership, alpha=0.3, color='#667eea', 
                       label='Inference Result')
    
    ax.set_xlabel('Attrition Risk Score (0-100)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Membership Degree', fontsize=11, fontweight='bold')
    ax.set_title('Output Membership Function dengan Prediction Results', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig


def get_rule_explanations_bahasa(activated_rules):
    """Explanations untuk activated rules dalam bahasa Indonesia"""
    
    rule_descriptions = {
        'R1': 'Kepuasan kerja RENDAH + Work-life balance RENDAH → SANGAT TINGGI',
        'R2': 'Kepuasan kerja RENDAH + Overtime TINGGI → SANGAT TINGGI',
        'R3': 'Work-life balance RENDAH + Overtime TINGGI → TINGGI',
        'R4': 'Performa BURUK + Kepuasan kerja RENDAH → TINGGI',
        'R5': 'Overtime TINGGI + Jam kerja SANGAT TINGGI → SANGAT TINGGI',
        'R6': 'Kepuasan kerja SEDANG + Work-life balance SEDANG → SEDANG',
        'R7': 'Overtime SEDANG + Jam kerja TINGGI → SEDANG',
        'R8': 'Kepuasan kerja SEDANG + Overtime SEDANG → SEDANG',
        'R9': 'Performa BIASA + Work-life balance SEDANG → SEDANG',
        'R10': 'Kepuasan kerja TINGGI + Work-life balance TINGGI → SANGAT RENDAH',
        'R11': 'Kepuasan kerja TINGGI + Performa BAGUS → RENDAH',
        'R12': 'Work-life balance TINGGI + Overtime RENDAH → RENDAH',
        'R13': 'Performa BAGUS + Overtime RENDAH → SANGAT RENDAH',
        'R14': 'Kepuasan kerja RENDAH + Performa BAGUS → SEDANG',
        'R15': 'Overtime RENDAH + Jam kerja NORMAL → RENDAH',
        'R16': 'Work-life balance RENDAH + Performa BURUK → TINGGI',
        'R17': 'Kepuasan kerja TINGGI + Overtime TINGGI → SEDANG'
    }
    
    explanations = []
    if activated_rules:
        sorted_rules = sorted(activated_rules, key=lambda x: x['firing_strength'], reverse=True)
        for rule in sorted_rules[:5]:
            rule_id = rule['rule_id']
            fs = rule['firing_strength']
            desc = rule_descriptions.get(rule_id, f'Rule {rule_id}')
            explanations.append({
                'Rule': rule_id,
                'Strength': f'{fs:.3f}',
                'IF-THEN': desc
            })
    
    return pd.DataFrame(explanations)


def create_sensitivity_analysis(mamdani, input_data):
    """What-if analysis: impact of changing each variable"""
    results = []
    baseline_mam, _, _ = mamdani.predict(input_data)
    
    variables_info = [
        ('JobSatisfaction', 1, 10, -1),
        ('WorkLifeBalance', 1, 10, -1),
        ('Overtime', 0, 10, 1),
        ('PerformanceRating', 1, 5, -0.5),
        ('AvgHoursPerWeek', 20, 80, 5)
    ]
    
    for var_name, var_min, var_max, change_amount in variables_info:
        current_val = input_data[var_name]
        
        # Test positive change
        test_input_pos = input_data.copy()
        test_input_pos[var_name] = min(var_max, current_val + change_amount)
        pred_pos, _, _ = mamdani.predict(test_input_pos)
        impact_pos = pred_pos - baseline_mam
        
        # Test negative change
        test_input_neg = input_data.copy()
        test_input_neg[var_name] = max(var_min, current_val - change_amount)
        pred_neg, _, _ = mamdani.predict(test_input_neg)
        impact_neg = baseline_mam - pred_neg
        
        results.append({
            'Variabel': var_name,
            'Nilai': f'{current_val:.1f}',
            'Naikkan': f'{impact_pos:+.2f}%',
            'Turunkan': f'{impact_neg:+.2f}%',
            'Sensitivitas': f'{(abs(impact_pos) + abs(impact_neg))/2:.2f}%'
        })
    
    return pd.DataFrame(results)


def init_session_state():
    """Initialize session state untuk tracking history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'history_count' not in st.session_state:
        st.session_state.history_count = 0


# ============= CUSTOM CSS STYLING =============

# ============= CUSTOM CSS STYLING - PREMIUM UI =============

st.markdown("""
<style>
    /* ===== MAIN PAGE STYLING ===== */
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stContainer {
        background: linear-gradient(to bottom, #f8f9ff 0%, #f0f2ff 100%);
    }
    
    /* ===== HEADER & TITLE STYLING ===== */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.8em;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    
    h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 1.8em;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    h3 {
        color: #667eea;
        font-weight: 700;
        font-size: 1.3em;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* ===== METRIC BOXES - PREMIUM DESIGN ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6);
    }
    
    .metric-value {
        font-size: 48px;
        margin: 10px 0;
        font-weight: 900;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.95;
        letter-spacing: 0.5px;
    }
    
    /* ===== RISK LEVEL BOXES ===== */
    .risk-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white;
    }
    
    .risk-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
    }
    
    /* ===== SLIDERS STYLING ===== */
    .stSlider {
        padding: 15px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stSlider:hover {
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ===== BUTTON STYLING ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.5px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
        border-radius: 10px;
        padding: 12px 15px;
        font-weight: 700;
        color: #667eea;
        border: 1px solid #d8e0ff;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e8ebff 0%, #dce0ff 100%);
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== INPUT BOX STYLING ===== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ===== TAB STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(to right, #f8f9ff 0%, #f0f2ff 100%);
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        color: #666;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid transparent;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== DIVIDER STYLING ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }
    
    /* ===== CARD CONTAINERS ===== */
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        border-color: #667eea;
    }
    
    /* ===== INFO BOX STYLING ===== */
    .stInfo {
        background: linear-gradient(135deg, #e8f4f8 0%, #dff0f7 100%);
        border-left: 4px solid #0ea5e9;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3e8 0%, #fdf2e0 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #ecf9f0 0%, #e0f8e8 100%);
        border-left: 4px solid #10b981;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fcd5d5 100%);
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* ===== TEXT STYLING ===== */
    p {
        color: #555;
        line-height: 1.6;
        font-size: 15px;
    }
    
    strong {
        color: #667eea;
        font-weight: 700;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    .stSidebar {
        background: linear-gradient(180deg, #f8f9ff 0%, #f0f2ff 100%);
    }
    
    .stSidebar .stMarkdown {
        background: transparent;
    }
    
    /* ===== CODE BLOCK STYLING ===== */
    code {
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
        color: #667eea;
        border-radius: 5px;
        padding: 2px 6px;
        font-weight: 600;
    }
    
    /* ===== SUMMARY BOX ===== */
    .summary-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8ebff 100%);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    
    /* ===== CHART CONTAINER ===== */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 15px 0;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stMetric {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        h1 {
            font-size: 2em;
        }
        
        h2 {
            font-size: 1.4em;
        }
        
        .metric-value {
            font-size: 36px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============= MAIN APPLICATION =============

# Initialize session state
init_session_state()

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="margin-bottom: 10px;">🧠 Fuzzy Attrition Predictor</h1>
    <p style="font-size: 18px; color: #667eea; font-weight: 600; margin: 0;">
        Prediksi Risiko Karyawan Menggunakan Fuzzy Logic Mamdani & Sugeno
    </p>
    <p style="font-size: 14px; color: #888; margin-top: 8px;">
        Implementasi 100% dari Scratch dengan 17 Fuzzy Rules
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Fuzzy Systems
mamdani, sugeno, variables = initialize_fuzzy_systems()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", 
                                   "📚 Methodology", "ℹ️ Info"])

# ============= SIDEBAR SECTION =============

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 12px; text-align: center;
                margin-bottom: 20px;">
        <h3 style="color: white; margin-top: 0;">🧠 Fuzzy Logic</h3>
        <p style="color: rgba(255, 255, 255, 0.9); margin: 5px 0; font-size: 13px;">
            Mamdani & Sugeno Methods
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Quick Settings")
    
    # Sidebar info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
                border-left: 4px solid #667eea;
                border-radius: 8px; padding: 15px; margin: 10px 0;">
        <h4 style="color: #667eea; margin-top: 0;">📊 About This Tool</h4>
        <p style="font-size: 13px; color: #555; margin: 8px 0;">
            Prediksi attrition risk menggunakan 17 fuzzy rules dengan 5 input variables.
        </p>
        <ul style="font-size: 12px; color: #666; margin: 8px 0; padding-left: 20px;">
            <li>Mamdani: Centroid defuzzification</li>
            <li>Sugeno: Weighted average</li>
            <li>100% from scratch implementation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============= TAB 1: SINGLE PREDICTION =============

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
                border-left: 4px solid #667eea;
                border-radius: 8px; padding: 15px; margin-bottom: 25px;">
        <p style="color: #667eea; font-weight: 600; margin: 0; font-size: 15px;">
            💡 Masukkan data karyawan di bawah untuk memprediksi attrition risk
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Input Data Karyawan")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("### 📋 Employee Factors")
        
        js = st.slider(
            "Job Satisfaction (Kepuasan Kerja)",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            help="1 = Sangat Tidak Puas | 10 = Sangat Puas"
        )
        
        wlb = st.slider(
            "Work-Life Balance",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            help="1 = Buruk | 10 = Sempurna"
        )
        
        ot = st.slider(
            "Overtime Hours (Per Minggu)",
            min_value=0.0, max_value=10.0, value=5.0, step=0.5,
            help="0 = Tidak ada | 10 = Sangat Banyak"
        )
        
        pr = st.slider(
            "Performance Rating",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5,
            help="1 = Buruk | 5 = Sangat Bagus"
        )
        
        ah = st.slider(
            "Avg Hours Per Week",
            min_value=20.0, max_value=80.0, value=45.0, step=1.0,
            help="20-80 jam per minggu"
        )
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
                    border-left: 4px solid #667eea;
                    border-radius: 8px; padding: 15px; margin-bottom: 15px;">
            <h4 style="color: #667eea; margin-top: 0; font-size: 16px;">📊 Input Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        
        summary_df = pd.DataFrame({
            "Faktor": ["Job Satisfaction", "Work-Life Balance", "Overtime", 
                      "Performance Rating", "Avg Hours/Week"],
            "Nilai": [f"{js:.1f}/10", f"{wlb:.1f}/10", f"{ot:.1f}/10", 
                     f"{pr:.1f}/5", f"{ah:.0f}"],
            "Status": [
                "✅ Baik" if js >= 7 else "⚠️ Sedang" if js >= 4 else "❌ Buruk",
                "✅ Baik" if wlb >= 7 else "⚠️ Sedang" if wlb >= 4 else "❌ Buruk",
                "✅ Baik" if ot <= 3 else "⚠️ Sedang" if ot <= 6 else "❌ Buruk",
                "✅ Baik" if pr >= 3.5 else "⚠️ Sedang" if pr >= 2 else "❌ Buruk",
                "✅ Normal" if 35 <= ah <= 50 else "⚠️ Tinggi" if ah <= 35 else "❌ SangatTinggi"
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Prediction
    input_data = {
        'JobSatisfaction': js,
        'WorkLifeBalance': wlb,
        'Overtime': ot,
        'PerformanceRating': pr,
        'AvgHoursPerWeek': ah
    }
    
    # Mamdani Prediction
    mamdani_score, mam_fuzzified, mam_rules = mamdani.predict(input_data)
    
    # Sugeno Prediction
    sugeno_score, seg_fuzzified, seg_rules = sugeno.predict(input_data)
    
    # Display Results
    st.markdown("## 🎯 Hasil Prediksi")
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 30px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <div style="font-size: 13px; opacity: 0.95; letter-spacing: 0.5px; font-weight: 600;">🔴 MAMDANI METHOD</div>
            <div style="font-size: 48px; font-weight: 900; margin: 15px 0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">{mamdani_score:.1f}%</div>
            <div style="font-size: 12px; opacity: 0.9;">Centroid (Center of Gravity)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    color: white; padding: 30px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 30px rgba(245, 87, 108, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <div style="font-size: 13px; opacity: 0.95; letter-spacing: 0.5px; font-weight: 600;">🟠 SUGENO METHOD</div>
            <div style="font-size: 48px; font-weight: 900; margin: 15px 0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">{sugeno_score:.1f}%</div>
            <div style="font-size: 12px; opacity: 0.9;">Weighted Average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diff = abs(mamdani_score - sugeno_score)
        corr = 100 - (diff / 100) * 100
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 30px; border-radius: 15px; text-align: center;
                    box-shadow: 0 8px 30px rgba(79, 172, 254, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;">
            <div style="font-size: 13px; opacity: 0.95; letter-spacing: 0.5px; font-weight: 600;">🔀 COMPARISON</div>
            <div style="font-size: 48px; font-weight: 900; margin: 15px 0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">{diff:.1f}</div>
            <div style="font-size: 12px; opacity: 0.9;">MAE | Correlation: {corr:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Levels
    st.markdown("## 📊 Risk Level Assessment")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        risk_level_m, _, color_m = get_risk_level(mamdani_score)
        st.markdown(f"""
        <div style="background: {color_m}; color: white; padding: 25px; 
                    border-radius: 15px; text-align: center; font-weight: bold; font-size: 19px;
                    box-shadow: 0 8px 25px {color_m}40;
                    border: 2px solid rgba(255, 255, 255, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;">
            {risk_level_m}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level_s, _, color_s = get_risk_level(sugeno_score)
        st.markdown(f"""
        <div style="background: {color_s}; color: white; padding: 25px; 
                    border-radius: 15px; text-align: center; font-weight: bold; font-size: 19px;
                    box-shadow: 0 8px 25px {color_s}40;
                    border: 2px solid rgba(255, 255, 255, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;">
            {risk_level_s}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 📈 Detailed Analysis")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("### Fuzzifikasi Inputs")
        fig_fuzz = plot_fuzzified_inputs(mam_fuzzified, variables)
        st.pyplot(fig_fuzz, use_container_width=True)
    
    with col2:
        st.markdown("### Activated Fuzzy Rules (Mamdani)")
        fig_rules = plot_activated_rules(mam_rules)
        if fig_rules:
            st.pyplot(fig_rules, use_container_width=True)
        else:
            st.info("Tidak ada rules yang teraktifkan")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("## 💡 Recommendations & Actions")
    
    recs = get_recommendations(mamdani_score, js, wlb, ot, pr, ah)
    
    # Create recommendation cards
    for i, rec in enumerate(recs, 1):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
                    border-left: 4px solid #667eea;
                    border-radius: 8px; padding: 12px 15px; margin: 10px 0;">
            <p style="margin: 0; color: #333; font-size: 14px;">
                <strong style="color: #667eea;">Rec {i}:</strong> {rec}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NEW: Output Membership Visualization
    st.markdown("## 📉 Output Membership Function")
    st.markdown("**Visualisasi bagaimana output risk score berasal dari activated rules:**")
    fig_output = plot_output_membership(variables['attrition_risk'], mamdani_score, 
                                        sugeno_score, mam_rules)
    st.pyplot(fig_output, use_container_width=True)
    
    st.markdown("---")
    
    # NEW: Rule Explanations
    st.markdown("## 📋 Penjelasan Aturan Fuzzy yang Aktif")
    st.markdown("**Top-5 rules dengan firing strength tertinggi:**")
    rule_exp_df = get_rule_explanations_bahasa(mam_rules)
    if not rule_exp_df.empty:
        st.dataframe(rule_exp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada rules yang teraktifkan")
    
    st.markdown("---")
    
    # NEW: Sensitivity Analysis
    with st.expander("🔍 Sensitivity Analysis (What-If Scenarios)"):
        st.markdown("**Analisis dampak perubahan setiap variabel terhadap attrition risk:**")
        sensitivity_df = create_sensitivity_analysis(mamdani, input_data)
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Interpretasi:**
        - **Sensitivitas tinggi** = Variabel berpengaruh besar terhadap attrition risk
        - **Naikkan** = Impact jika variabel dinaikkan
        - **Turunkan** = Impact jika variabel diturunkan
        """)
    
    st.markdown("---")
    
    # NEW: Add to History
    history_entry = {
        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
        'JS': f'{js:.1f}',
        'WLB': f'{wlb:.1f}',
        'OT': f'{ot:.1f}',
        'PR': f'{pr:.1f}',
        'AH': f'{ah:.0f}',
        'Mamdani': f'{mamdani_score:.1f}%',
        'Sugeno': f'{sugeno_score:.1f}%'
    }
    st.session_state.prediction_history.append(history_entry)
    
    # Show prediction history in sidebar
    with st.sidebar.expander("📝 Prediction History", expanded=False):
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history[-10:])  # Last 10
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("Belum ada prediction history")
    
    st.markdown("---")
    
    # Comparison Table
    st.markdown("## 🔄 Mamdani vs Sugeno Comparison")
    
    comparison_df = pd.DataFrame({
        "Aspek": [
            "Defuzzification Method",
            "Output Type",
            "Computation Speed",
            "Interpretability",
            "Scalability",
            "Implementation",
            "Best For"
        ],
        "Mamdani": [
            "Centroid (CoG)",
            "Fuzzy Set → Crisp",
            "Slower (numerical integration)",
            "Very High (linguistic)",
            "Medium",
            "More Complex",
            "Decision Support Systems"
        ],
        "Sugeno": [
            "Weighted Average",
            "Direct Crisp",
            "Faster (arithmetic)",
            "Lower (numeric)",
            "High",
            "Simpler",
            "Real-time Systems"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============= TAB 2: BATCH PREDICTION =============

with tab2:
    st.markdown("## 📦 Batch Prediction")
    
    st.markdown("Upload CSV file dengan kolom: JobSatisfaction, WorkLifeBalance, Overtime, PerformanceRating, AvgHoursPerWeek")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            st.markdown(f"### Preview Data ({len(df_batch)} records)")
            st.dataframe(df_batch.head(), use_container_width=True)
            
            if st.button("🚀 Predict Batch"):
                progress_bar = st.progress(0)
                
                predictions = []
                for idx, row in df_batch.iterrows():
                    input_dict = {
                        'JobSatisfaction': row['JobSatisfaction'],
                        'WorkLifeBalance': row['WorkLifeBalance'],
                        'Overtime': row['Overtime'],
                        'PerformanceRating': row['PerformanceRating'],
                        'AvgHoursPerWeek': row['AvgHoursPerWeek']
                    }
                    
                    mam_score, _, _ = mamdani.predict(input_dict)
                    seg_score, _, _ = sugeno.predict(input_dict)
                    
                    mam_level, _, _ = get_risk_level(mam_score)
                    seg_level, _, _ = get_risk_level(seg_score)
                    
                    predictions.append({
                        'Index': idx + 1,
                        'Mamdani Score': round(mam_score, 2),
                        'Mamdani Level': mam_level,
                        'Sugeno Score': round(seg_score, 2),
                        'Sugeno Level': seg_level
                    })
                    
                    progress_bar.progress((idx + 1) / len(df_batch))
                
                st.success("✅ Batch prediction complete!")
                
                results_df = pd.DataFrame(predictions)
                st.markdown("### Hasil Prediksi")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Export to CSV
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_buffer,
                    file_name="attrition_predictions.csv",
                    mime="text/csv"
                )
                
                # Statistics
                st.markdown("### 📊 Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(results_df))
                
                with col2:
                    avg_mam = results_df['Mamdani Score'].mean()
                    st.metric("Avg Mamdani Score", f"{avg_mam:.1f}%")
                
                with col3:
                    avg_seg = results_df['Sugeno Score'].mean()
                    st.metric("Avg Sugeno Score", f"{avg_seg:.1f}%")
                
                with col4:
                    corr = results_df['Mamdani Score'].corr(results_df['Sugeno Score'])
                    st.metric("Correlation", f"{corr:.3f}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============= TAB 3: METHODOLOGY =============

with tab3:
    st.markdown("## 📚 Methodology")
    
    st.markdown("""
    ### Implementasi Fuzzy Logic dari Scratch
    
    Sistem ini mengimplementasikan Fuzzy Logic 100% dari scratch tanpa menggunakan library fuzzy eksternal 
    (seperti scikit-fuzzy). Setiap komponen dibangun dari dasar dengan Python pure.
    
    #### 1. **Triangular Membership Function**
    
    Fungsi keanggotaan segitiga dengan parameter (a, b, c):
    - a: Lower bound
    - b: Peak (puncak)
    - c: Upper bound
    
    #### 2. **Fuzzification Process**
    
    Konversi nilai crisp input menjadi derajat keanggotaan fuzzy untuk setiap linguistic term.
    
    Contoh: JobSatisfaction = 7
    - Rendah: 0.0
    - Sedang: 0.4
    - Tinggi: 0.6
    
    #### 3. **Inference Engine (17 Fuzzy Rules)**
    
    Evaluasi 17 if-then rules menggunakan operator MIN untuk AND logic:
    
    - **Rules 1-5**: High Risk (Score 70-90)
    - **Rules 6-9**: Medium Risk (Score 48-55)
    - **Rules 10-13**: Low Risk (Score 10-35)
    - **Rules 14-17**: Mixed Scenarios (Score 45-72)
    
    #### 4. **Defuzzification Methods**
    
    **A. Mamdani Method (Centroid)**
    ```
    z* = ∫ z·μ(z) dz / ∫ μ(z) dz
    ```
    Output berupa fuzzy set yang di-convert menjadi crisp value menggunakan center of gravity.
    
    **B. Sugeno Method (Zero-Order, Weighted Average)**
    ```
    z* = Σ(αᵢ × zᵢ) / Σ(αᵢ)
    ```
    Output langsung berupa nilai crisp dari kombinasi weighted dari rule outputs.
    
    #### 5. **5 Input Variables**
    
    | Variable | Domain | Linguistic Terms |
    |---|---|---|
    | JobSatisfaction | 1-10 | Rendah, Sedang, Tinggi |
    | WorkLifeBalance | 1-10 | Rendah, Sedang, Tinggi |
    | Overtime | 0-10 | Rendah, Sedang, Tinggi |
    | PerformanceRating | 1-5 | Buruk, Biasa, Bagus |
    | AvgHoursPerWeek | 20-80 | Normal, Tinggi, SangatTinggi |
    
    #### 6. **1 Output Variable**
    
    | Variable | Domain | Linguistic Terms |
    |---|---|---|
    | AttritionRisk | 0-100 | SangatRendah, Rendah, Sedang, Tinggi, SangatTinggi |
    
    ### Python Classes Structure
    
    - `TriangularMembership`: Implementasi fungsi keanggotaan segitiga
    - `FuzzyVariable`: Container untuk multiple fuzzy sets
    - `FuzzyRule`: IF-THEN rule dengan conditions & conclusion
    - `FuzzyMamdani`: Sistem Mamdani lengkap dengan centroid defuzzification
    - `FuzzySugeno`: Sistem Sugeno lengkap dengan weighted average
    """)

# ============= TAB 4: INFO =============

with tab4:
    st.markdown("## ℹ️ Aplikasi Information")
    
    st.markdown("""
    ### 📋 Project Info
    
    - **Nama Proyek**: TUBES DKA - Dasar Kecerdasan Artifisial
    - **Institusi**: Telkom University Purwokerto
    - **Topik**: Fuzzy Logic untuk Employee Attrition Prediction
    - **Metode**: Mamdani vs Sugeno Comparison
    - **Dataset**: Kaggle Employee Attrition (10,000 records)
    
    ### 👥 Kelompok Rio
    
    1. Tri Setyono Martyantoro (101132400279)
    2. Nanda Bagus Priambodo (101132430007)
    3. Abyan Rahman Al Fariz (101132430021)
    
    ### 📊 Key Statistics
    
    - **Input Variables**: 5 (Continuous)
    - **Output Variable**: 1 (Attrition Risk: 0-100)
    - **Fuzzy Rules**: 17 IF-THEN rules
    - **Linguistic Terms**: 18 total (across all variables)
    - **Implementation**: 100% from scratch (no external fuzzy libraries)
    
    ### 🔧 Technical Stack
    
    - **Language**: Python 3.x
    - **Framework**: Streamlit
    - **Libraries**: NumPy, Pandas, Matplotlib, SciPy
    - **Computation**: Numerical integration for centroid calculation
    
    ### 📚 Related Files
    
    - `fuzzy_analysis_colab.ipynb`: Main research & implementation (41 cells, 2000+ lines)
    - `employee_attrition_dataset_10000.csv`: Dataset (10,000 records)
    - `streamlit_app_improved.py`: Interactive web application (this file)
    - `README.md`: Project documentation
    - `PANDUAN_LENGKAP.md`: Technical guide
    - `DELIVERABLES.md`: Requirements checklist
    
    ### 🎯 Features
    
    ✅ Real-time prediction dengan Fuzzy Mamdani  
    ✅ Real-time prediction dengan Fuzzy Sugeno  
    ✅ Visualisasi fuzzifikasi inputs  
    ✅ Tampilan activated rules  
    ✅ Batch prediction dari file CSV  
    ✅ Export hasil prediksi  
    ✅ Detailed recommendations  
    ✅ Methodology explanation  
    
    ---
    
    **Version**: 3.0 (Production Ready)  
    **Last Updated**: June 2026  
    **License**: Academic Project (DKA - Telkom University)
    """)

# ============= FOOTER =============

st.markdown("""
---

<div style='text-align: center; color: #888; margin-top: 30px;'>
    <p><strong>Streamlit Application v3 - Production Ready</strong></p>
    <p>Implementasi Fuzzy Logic Mamdani vs Sugeno untuk Employee Attrition Prediction</p>
    <p>TUBES DKA | Telkom University Purwokerto | 2026</p>
</div>
""", unsafe_allow_html=True)
