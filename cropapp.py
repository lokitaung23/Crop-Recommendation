# cropapp.py — RF-as-initial-policy + Contextual Bandit RL
# Tabs + Farm details + safe input reset + STICKY recommendation panel
# Supabase authentication + role-aware session

import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
import subprocess, sys

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(
    page_title="Crop Advisor — Kenya Central Highlands",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================
# Load env + Supabase client
# =========================================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
TABLE = "predictions"

# ---------- Role helper ----------
def fetch_role_for_email(email: str) -> str:
    """Fetch the role for a given email from the user_profiles table."""
    try:
        res = sb.table("user_profiles").select("role").eq("email", email).single().execute()
        data = res.data or {}
        role = data.get("role")
        if role in ("Admin", "Farmer", "Extension officer"):
            return role
        return "Farmer"  # fallback
    except Exception:
        return "Farmer"

# -------- Admin role-management helpers --------
ALLOWED_ROLES = ("Admin", "Farmer", "Extension officer")

def list_user_profiles() -> pd.DataFrame:
    try:
        res = sb.table("user_profiles").select("email, role, created_at, updated_at").order("email").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.error(f"Error reading user_profiles: {e}")
        return pd.DataFrame(columns=["email","role","created_at","updated_at"])

def upsert_user_role(email: str, role: str) -> bool:
    if not email or role not in ALLOWED_ROLES:
        st.warning("Provide a valid email and role.")
        return False
    try:
        sb.table("user_profiles").upsert({"email": email, "role": role}).execute()
        return True
    except Exception as e:
        st.error(f"Error updating role: {e}")
        return False

def refresh_role_from_db():
    """Refresh role in session from user_profiles (handles role changes without re-login)."""
    user = st.session_state.get("auth_user")
    if not user or not user.get("email"):
        return
    try:
        new_role = fetch_role_for_email(user["email"])
        if new_role and new_role != user.get("role"):
            st.session_state["auth_user"]["role"] = new_role
    except Exception:
        pass

# ---------- Auth helpers ----------
def _auth_get_user():
    """Return cached user if available, otherwise None."""
    return st.session_state.get("auth_user")

def _auth_set_session(session):
    """Store auth session with resolved role from Supabase."""
    email = session.user.email if session and session.user else None
    role = fetch_role_for_email(email) if email else "Farmer"
    st.session_state["auth_user"] = {
        "email": email,
        "id": session.user.id if session and session.user else None,
        "role": role,
        "access_token": getattr(session, "access_token", None),
    }

def _auth_clear():
    st.session_state.pop("auth_user", None)
    try:
        sb.auth.sign_out()
    except Exception:
        pass

def login_ui_supabase():
    """Render a blocking login form."""
    st.title("🔐 Sign in to continue")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", key="login_email", autocomplete="username")
        password = st.text_input("Password", type="password", key="login_password", autocomplete="current-password")
        col1, col2 = st.columns([1,1])
        with col1:
            submit = st.form_submit_button("Sign in", use_container_width=True)
        with col2:
            signup = st.form_submit_button("Create account", use_container_width=True)

    if signup:
        if not email or not password:
            st.warning("Enter email & password to create an account.")
            st.stop()
        try:
            sb.auth.sign_up({"email": email, "password": password})
            sb.table("user_profiles").upsert({"email": email, "role": "Farmer"}).execute()
            st.success("Account created. Check your email for confirmation (if enabled), then sign in.")
        except Exception as e:
            st.error(f"Sign-up failed: {e}")
        st.stop()

    if submit:
        if not email or not password:
            st.warning("Email and password are required.")
            st.stop()
        try:
            session = sb.auth.sign_in_with_password({"email": email, "password": password})
            _auth_set_session(session)
            st.toast("Signed in successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Sign-in failed: {e}")
            st.stop()

def ensure_logged_in():
    """Block page until user is logged in."""
    user = _auth_get_user()
    if not user or not user.get("email"):
        login_ui_supabase()
        st.stop()

def logout_button():
    with st.sidebar:
        st.markdown("### Crop Advisor")
        user = _auth_get_user()
        role = (user or {}).get("role","Farmer")
        email = (user or {}).get("email","—")

        # user card
        with st.container():
            st.markdown(f"""
<div class="card soft">
  <div><b>Signed in:</b><br>{email}</div>
  <div style="margin-top:.4rem"><b>Role:</b> {role}</div>
</div>
""", unsafe_allow_html=True)

        cols = st.columns([1,1])
        with cols[0]:
            if st.button("Logout", use_container_width=True):
                _auth_clear()
                st.rerun()
        with cols[1]:
            if st.button("Refresh role", use_container_width=True):
                refresh_role_from_db()
                st.toast("Role refreshed")
                st.rerun()

        st.markdown('<div class="footer">Farmer-centered. Feedback-powered. Continuously learning.</div>', unsafe_allow_html=True)


# Optional (only if you might have a CNN fallback)
try:
    from tensorflow.keras.models import load_model  # noqa: F401
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# =========================
# Reset machinery (must run BEFORE widgets are created)
# =========================
def _reset_recommend_form():
    for k, default in [
        ("inp_N", 0.0),
        ("inp_P", 0.0),
        ("inp_K", 0.0),
        ("inp_temp", 0.0),
        ("inp_humidity", 0.0),
        ("inp_ph", 0.0),
        ("inp_rainfall", 0.0),
    ]:
        st.session_state[k] = default

def _reset_feedback_form():
    for k, default in [
        ("fb_farm_search", ""),
        ("fb_visit_time", 0),
        ("fb_radio", "Good"),
    ]:
        st.session_state[k] = default

if "do_reset_reco" not in st.session_state:
    st.session_state.do_reset_reco = False
if "do_reset_feedback" not in st.session_state:
    st.session_state.do_reset_feedback = False
if "last_reco" not in st.session_state:
    st.session_state.last_reco = None

if st.session_state.do_reset_reco:
    _reset_recommend_form()
    st.session_state.do_reset_reco = False

if st.session_state.do_reset_feedback:
    _reset_feedback_form()
    st.session_state.do_reset_feedback = False

# =========================
# Reset machinery (must run BEFORE widgets are created)
# =========================
# ... your reset helpers ...

import base64, os

def _logo_data_uri():
    # Try a few common locations
    for p in ("logo.png", ".streamlit/static/logo.png", "app/static/logo.png"):
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    return None

LOGO_SRC = _logo_data_uri()

# ---- Header Banner (HTML + CSS) ----
st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --card: #151823;
  --muted: #a3a3a3;
  --accent: #ec5b2a;
  --success: #2ecc71;
  --radius: 16px;
}

.block-container { padding-top: 1.2rem; max-width: 1200px; }

.header-card {
  display:flex; align-items:center; background:var(--card);
  border-radius:var(--radius); padding:16px 22px; margin-bottom:14px;
  box-shadow:0 4px 16px rgba(0,0,0,.25); border:1px solid rgba(255,255,255,.06);
}
.header-logo {flex:0 0 auto;margin-right:18px;}
.header-title {flex:1 1 auto;}
.header-title h1 {font-size:26px;margin:0;color:#fff; letter-spacing:.2px;}
.header-title p  {margin:4px 0 0;font-size:13px;color:var(--muted);}

.card {
  background: var(--card);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--radius);
  padding: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.soft { border-radius: 14px; }

.stTabs [data-baseweb="tab-list"] { gap: 12px; }
.stTabs [data-baseweb="tab"] {
  background: var(--card); padding: 10px 14px; border-radius: 12px;
  border: 1px solid rgba(255,255,255,.08);
}
.stTabs [aria-selected="true"] { border-color: var(--accent); }

div[data-testid="stForm"] .stButton>button {
  width: 100%; height: 42px; border-radius: 12px;
}

.kpi { display:flex; gap:12px; }
.kpi .pill {
  background:#1c2333; border:1px solid rgba(255,255,255,.06);
  padding:8px 12px; border-radius:999px; color:#fff; font-size:13px;
}

.stNumberInput input { border-radius:12px !important; }
.stTextInput input { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

logo_img = f'<img src="{LOGO_SRC}" width="100">' if LOGO_SRC else ""
st.markdown(f"""
<div class="header-card">
  <div class="header-logo">{logo_img}</div>
  <div class="header-title">
    <h1>Kenya Central Highlands Crop Advisory System</h1>
    <p>Farmer-centered. Feedback-powered. Continuously learning.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not LOGO_SRC:
    st.warning("Logo not found. Place 'logo.png' next to cropapp.py (or in .streamlit/static/).")


# ---- Auth Gate ----
ensure_logged_in()
logout_button()
refresh_role_from_db()



# =========================
# Load scaler & label encoder & model (after auth)
# =========================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
N_CLASSES = len(label_encoder.classes_)

is_cnn = False
try:
    with open("best_crop_model.pkl", "rb") as f:
        base_model = pickle.load(f)
except Exception:
    if TF_AVAILABLE and os.path.exists("best_crop_model_cnn.h5"):
        base_model = load_model("best_crop_model_cnn.h5")
        is_cnn = True
    else:
        raise RuntimeError("No base model found (best_crop_model.pkl or best_crop_model_cnn.h5).")

# =========================
# RL agent (contextual bandit)
# =========================
from rl_agent import ContextualBanditAgent
AGENT_PATH = "rl_agent.pkl"

if os.path.exists(AGENT_PATH):
    agent = ContextualBanditAgent.load(AGENT_PATH)
else:
    agent = ContextualBanditAgent(
        n_actions=N_CLASSES,
        feature_dim=7,
               epsilon=0.10,
        random_state=123
    )
    agent.save(AGENT_PATH)

# =========================
# Helpers
# =========================
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "Annual Rainfall"]

def to_scaled_df(N, P, K, temperature, humidity, ph, rainfall):
    df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURE_ORDER)
    return df, scaler.transform(df)

def base_policy_proba(model, X_scaled):
    if is_cnn:
        X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
        p = model.predict(X_reshaped, verbose=0)
        return np.asarray(p)
    else:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_scaled)
            return np.asarray(p)
        return np.ones((X_scaled.shape[0], N_CLASSES)) / N_CLASSES

def blended_scores(agent, X_scaled, base_probs):
    reward_preds = agent.predict_reward(X_scaled)  # (1, K)
    rp = reward_preds.copy()
    rp_min = rp.min(axis=1, keepdims=True)
    rp_max = rp.max(axis=1, keepdims=True)
    denom = np.where((rp_max - rp_min) == 0, 1.0, (rp_max - rp_min))
    rp_norm = (rp - rp_min) / denom
    blended = 0.6 * rp_norm + 0.4 * base_probs
    greedy = int(np.argmax(blended[0]))
    return blended, greedy

def epsilon_greedy_logging_prop(epsilon, greedy_action, chosen_action, n_actions):
    if chosen_action == greedy_action:
        return (1.0 - epsilon) + (epsilon / n_actions)
    else:
        return (epsilon / n_actions)

def load_history() -> pd.DataFrame:
    try:
        res = sb.table(TABLE).select("*").order("Timestamp").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.warning(f"Could not read from database: {e}")
        return pd.DataFrame()

def save_history(df: pd.DataFrame):
    try:
        payload = df.to_dict(orient="records")
        if payload:
            sb.table(TABLE).insert(payload).execute()
    except Exception as e:
        st.error(f"DB insert failed: {e}")

# ---- Role-based tabs ----
user = st.session_state.get("auth_user", {})
role = (user or {}).get("role", "Farmer")

if role == "Admin":
    tab_labels = ["📈 Reports", "🧪 Retrain Models"]
elif role == "Extension officer":
    tab_labels = ["🧮 Recommend", "📝 Feedback", "📈 Reports"]
else:  # Farmer
    tab_labels = ["🧮 Recommend", "📝 Feedback"]

tabs = st.tabs(tab_labels)
TAB = {label: tabs[i] for i, label in enumerate(tab_labels)}   # name -> tab object

# =========================================
# Recommend (Farmer + Officer)
# =========================================
if "🧮 Recommend" in TAB:
    with TAB["🧮 Recommend"]:
        st.subheader("Farmer Details")
        col_id, col_phone = st.columns([1, 1])
        with col_id:
            farm_number = st.text_input("Farm Number (unique)*", key="farm_number",
                                        help="This will identify the farm across visits.")
        with col_phone:
            phone = st.text_input("Phone Number", key="phone",
                                  help="Optional but helpful for follow-ups.")

        if role in ("Extension officer", "Admin") and farm_number.strip():
            st.caption(f"👩🏾‍🌾 Acting on behalf of **{farm_number.strip()}**")

        if st.session_state.last_reco:
            lr = st.session_state.last_reco
            st.markdown("### 🌾 Last recommendation")
            st.success(f"**{lr['chosen_crop']}** for Farm **{lr['farm_number']}** at {lr['timestamp']}")
            with st.expander("Base model Top-3 (reference):", expanded=False):
                for crop, prob in lr["top3"]:
                    st.write(f"- {crop}: {prob:.2%}")

        st.subheader("🔍 Enter Input Parameters")
        with st.form("input_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                N = st.number_input("Nitrogen (N)", min_value=0.0, key="inp_N")
                temperature = st.number_input("temperature (°C)", min_value=-10.0, max_value=60.0, key="inp_temp")
                ph = st.number_input("ph", min_value=0.0, max_value=14.0, key="inp_ph")
            with c2:
                P = st.number_input("Phosphorus (P)", min_value=0.0, key="inp_P")
                humidity = st.number_input("humidity (%)", min_value=0.0, max_value=100.0, key="inp_humidity")
                rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, key="inp_rainfall")
            with c3:
                K = st.number_input("Potassium (K)", min_value=0.0, key="inp_K")

            submitted = st.form_submit_button("Recommend")

        if submitted:
            if not farm_number.strip():
                st.error("Farm Number is required. Please enter a unique Farm Number.")
            else:
                input_df, X_scaled = to_scaled_df(N, P, K, temperature, humidity, ph, rainfall)

                base_probs = base_policy_proba(base_model, X_scaled)  # (1, K)

                actions, _ = agent.choose_action(X_scaled, base_policy_proba=base_probs)
                action_id = int(actions[0])
                chosen_crop = label_encoder.inverse_transform([action_id])[0]

                blended, greedy_action = blended_scores(agent, X_scaled, base_probs)
                logging_propensity = epsilon_greedy_logging_prop(agent.epsilon, greedy_action, action_id, N_CLASSES)
                base_prob_chosen = float(base_probs[0, action_id])

                base_probs_row = base_probs[0]
                top3_idx = base_probs_row.argsort()[-3:][::-1]
                top3_crops = label_encoder.inverse_transform(top3_idx)
                top3_probs = base_probs_row[top3_idx]
                top3_pairs = list(zip(top3_crops, [float(p) for p in top3_probs]))

                st.subheader("🌾 Recommended Crop (RL-blended):")
                st.markdown(f"### ✅ {chosen_crop}")

                st.subheader("📊 Base Model Top-3 (reference)")
                for crop, prob in top3_pairs:
                    st.write(f"- {crop}: {prob:.2%}")

                try:
                    hist = load_history()

                    rec = input_df.copy()
                    rec["FarmNumber"] = farm_number.strip()
                    rec["Phone"] = phone.strip()
                    rec["RL_Chosen_Crop"] = chosen_crop
                    rec["Base_Top1"] = top3_crops[0]
                    rec["Base_Top1_Prob"] = float(top3_probs[0])
                    rec["Base_Top2"] = top3_crops[1]
                    rec["Base_Top2_Prob"] = float(top3_probs[1])
                    rec["Base_Top3"] = top3_crops[2]
                    rec["Base_Top3_Prob"] = float(top3_probs[2])

                    rec["GreedyAction"] = greedy_action
                    rec["ChosenAction"] = action_id
                    rec["LoggingPropensity"] = float(logging_propensity)
                    rec["BaseProbChosen"] = float(base_prob_chosen)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    rec["Timestamp"] = timestamp

                    #auth_user = st.session_state.get("auth_user")
                    #if auth_user and auth_user.get("email"):
                        #rec["email"] = auth_user["email"]

                    if not hist.empty:
                        hist = pd.concat([hist, rec], ignore_index=True)
                    else:
                        hist = rec

                    save_history(rec)
                    st.success(f"Saved recommendation for Farm **{farm_number}**.")

                    st.session_state.last_reco = {
                        "farm_number": farm_number.strip(),
                        "timestamp": timestamp,
                        "chosen_crop": chosen_crop,
                        "top3": top3_pairs,
                    }

                    st.session_state.do_reset_reco = True
                    st.rerun()

                except Exception as e:
                    st.warning(f"⚠️ Failed to save record: {e}")

# =========================================
# Feedback (Farmer + Officer)
# =========================================
if "📝 Feedback" in TAB:
    with TAB["📝 Feedback"]:
        st.subheader("Find Recommendations by Farm Number")
        fb_farm = st.text_input("Enter Farm Number", key="fb_farm_search")

        if fb_farm:
            hist = load_history()
            if hist.empty or "FarmNumber" not in hist.columns:
                st.info("No recommendation history found yet.")
            else:
                subset = hist[hist["FarmNumber"].astype(str).str.strip() == fb_farm.strip()].copy()
                if subset.empty:
                    st.warning("No records found for that Farm Number.")
                else:
                    subset["Timestamp_dt"] = pd.to_datetime(subset["Timestamp"], errors="coerce")
                    subset = subset.sort_values("Timestamp_dt")
                    options = subset["Timestamp"].tolist()
                    default_idx = len(options) - 1 if options else 0
                    chosen_time = st.selectbox("Select visit time", options, index=default_idx, key="fb_visit_time")

                    row = subset[subset["Timestamp"] == chosen_time].iloc[0]
                    st.write("📄 Selected Record:")
                    st.dataframe(
                        row[FEATURE_ORDER + ["RL_Chosen_Crop", "Base_Top1", "Base_Top2", "Base_Top3"]]
                        .to_frame()
                        .rename(columns={0: "Value"})
                    )

                    feedback = st.radio("Was the recommendation accurate?", ["Good", "Not good"],
                                        horizontal=True, key="fb_radio")
                    if st.button("Submit Feedback", key="fb_submit"):
                        try:
                            reward = 1.0 if feedback == "Good" else 0.0

                            (
                                sb.table(TABLE)
                                .update({"Feedback": feedback, "Reward": reward})
                                .eq("FarmNumber", fb_farm.strip())
                                .eq("Timestamp", chosen_time)
                                .execute()
                            )

                            feat = {k: float(row[k]) for k in FEATURE_ORDER}
                            X_scal = scaler.transform(pd.DataFrame([feat]))
                            chosen_crop = str(row.get("RL_Chosen_Crop", row.get("Base_Top1")))
                            action_id = int(label_encoder.transform([chosen_crop])[0])

                            agent.update(X_scal, np.array([action_id]), np.array([reward], dtype=float))
                            agent.save(AGENT_PATH)

                            st.success("✅ Feedback saved and RL agent updated.")
                            st.session_state.do_reset_feedback = True
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error: {e}")

# =========================================
# Reports / Analytics (Officer + Admin)
# =========================================
if "📈 Reports" in TAB:
    with TAB["📈 Reports"]:
        st.subheader("📈 Prediction Reports & Analytics")

        # ---- Admin-only: Manage Users & Roles (if helpers present) ----
        if role == "Admin" and "list_user_profiles" in globals():
            with st.expander("🛡️ Manage Users & Roles (Admin only)", expanded=False):
                users_df = list_user_profiles()
                if users_df.empty:
                    st.info("No users in user_profiles yet.")
                else:
                    st.dataframe(users_df, use_container_width=True)

                st.markdown("### Update or Create a User Profile")
                colu1, colu2, colu3 = st.columns([2,1,1])
                with colu1:
                    target_email = st.text_input("User email")
                with colu2:
                    new_role = st.selectbox("Role", ALLOWED_ROLES, index=ALLOWED_ROLES.index("Farmer"))
                with colu3:
                    if st.button("Save role", type="primary", use_container_width=True):
                        ok = upsert_user_role(target_email.strip(), new_role)
                        if ok:
                            st.success(f"Saved: {target_email} → {new_role}")
                            st.rerun()

        # ---- Filters ----
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            farm_filter = st.text_input("Filter by Farm Number (optional)")
        with c2:
            date_from = st.date_input("From (optional)", value=None)
        with c3:
            date_to = st.date_input("To (optional)", value=None)

        # ---- Load data ----
        df = load_history()
        if df.empty:
            st.info("No data in the predictions table yet.")
            st.stop()

        # Always create Timestamp_dt safely (Series matching df length)
        if "Timestamp" in df.columns:
            df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        else:
            df["Timestamp_dt"] = pd.to_datetime(pd.Series([None] * len(df)))

        # ---- Apply filters ----
        if farm_filter:
            df = df[df["FarmNumber"].astype(str).str.strip() == farm_filter.strip()]
        if date_from is not None:
            df = df[df["Timestamp_dt"] >= pd.to_datetime(date_from)]
        if date_to is not None:
            df = df[df["Timestamp_dt"] < (pd.to_datetime(date_to) + pd.Timedelta(days=1))]

        if df.empty:
            st.warning("No rows match your current filters.")
            st.stop()

        # Derive Reward from Feedback if needed
        if "Reward" not in df.columns and "Feedback" in df.columns:
            df["Reward"] = df["Feedback"].map({"Good": 1.0, "Not good": 0.0})

        # ---- KPIs ----
        total = len(df)
        with_feedback = df["Reward"].notna().sum() if "Reward" in df.columns else 0
        avg_reward = float(df["Reward"].mean()) if "Reward" in df.columns else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total predictions", f"{total:,}")
        m2.metric("Rows with feedback", f"{with_feedback:,}")
        m3.metric("Avg reward (accuracy)", f"{avg_reward:.2f}" if not pd.isna(avg_reward) else "—")

        st.divider()

# ---- Top crops ----
chosen_col = "RL_Chosen_Crop" if "RL_Chosen_Crop" in df.columns else "Base_Top1"
st.subheader("🏆 Top Crops (by selections)")

sel_counts = (
    df[chosen_col]
      .value_counts()
      .rename_axis("Crop")
      .reset_index(name="Selections")
)

# Add 1-based serial numbers
sel_counts.index = sel_counts.index + 1
sel_counts = sel_counts.rename_axis("No.").reset_index()

st.dataframe(sel_counts, use_container_width=True)
if not sel_counts.empty:
    st.bar_chart(sel_counts.set_index("Crop")[["Selections"]])


        # ---- Success rate per crop ----
        if "Reward" in df.columns and df["Reward"].notna().any():
            st.subheader("✅ Success rate per crop")
            success = (
                df.dropna(subset=["Reward"])
                  .groupby(chosen_col)["Reward"]
                  .mean()
                  .sort_values(ascending=False)
                  .rename("SuccessRate")
                  .reset_index()
            )
            st.dataframe(success, use_container_width=True)
            if not success.empty:
                st.bar_chart(success.set_index(chosen_col))

        # ---- Weekly trends ----
        st.subheader("📅 Weekly trend")
        tmp = df.copy()
        if "Timestamp_dt" not in tmp.columns:
            tmp["Timestamp_dt"] = pd.to_datetime(tmp.get("Timestamp"), errors="coerce")
        tmp = tmp.set_index("Timestamp_dt").sort_index()
        if tmp.index.notna().any():
            weekly = tmp.resample("W").agg({
                "Reward": "mean" if "Reward" in tmp.columns else "sum",
                chosen_col: "count"
            }).rename(columns={chosen_col: "Predictions"})
            if not weekly.empty:
                st.line_chart(weekly[["Predictions"]])
                if "Reward" in tmp.columns:
                    st.line_chart(weekly[["Reward"]])
            else:
                st.info("Not enough dated data to plot weekly trend.")
        else:
            st.info("No valid timestamps available to plot trends.")

        st.divider()

        # ---- Filtered rows (defensive handling) ----
        st.subheader("🔎 Filtered rows")
        show_cols = [c for c in [
            "Timestamp", "FarmNumber", "Phone", chosen_col,
            "Base_Top1","Base_Top2","Base_Top3",
            "Reward","Feedback","LoggingPropensity","BaseProbChosen","email"
        ] if c in df.columns]

        if "Timestamp_dt" not in df.columns:
            if "Timestamp" in df.columns:
                df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            else:
                df["Timestamp_dt"] = pd.to_datetime(pd.Series([None] * len(df)))

        try:
            df_sorted = df.sort_values("Timestamp_dt", na_position="last")
        except Exception:
            df_sorted = df

        st.dataframe(df_sorted[show_cols], use_container_width=True)

        @st.cache_data
        def to_csv_bytes(frame: pd.DataFrame) -> bytes:
            return frame.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Export filtered CSV",
            data=to_csv_bytes(df_sorted[show_cols]),
            file_name="predictions_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================================
# Retrain Models (Admin only)
# =========================================
if "🧪 Retrain Models" in TAB:
    with TAB["🧪 Retrain Models"]:
        st.subheader("🧪 Retrain Models")

        st.markdown(
            "This will run your local training script (e.g. `Models_Combined.py`) "
            "which should read your dataset (e.g. `MERGED.csv`) and write updated "
            "`best_crop_model.pkl`, `scaler.pkl`, and `label_encoder.pkl`."
        )

        script_path = "Models_Combined.py"   # change if your trainer has a different name

        if not os.path.exists(script_path):
            st.error(f"Training script not found: `{script_path}`")
        else:
            run = st.button("Start retraining", type="primary")
            if run:
                with st.spinner("Training… please wait"):
                    try:
                        proc = subprocess.run([sys.executable, script_path],
                                              capture_output=True, text=True, check=False)
                        st.code(proc.stdout or "(no stdout)", language="bash")
                        if proc.returncode != 0:
                            st.error("Training failed.")
                            st.code(proc.stderr or "(no stderr)", language="bash")
                        else:
                            st.success("Training finished. Reloading model artifacts…")
                            # top-level reassignment; no global needed
                            is_cnn = False
                            with open("best_crop_model.pkl", "rb") as f:
                                base_model = pickle.load(f)
                            with open("scaler.pkl", "rb") as f:
                                scaler = pickle.load(f)
                            with open("label_encoder.pkl", "rb") as f:
                                label_encoder = pickle.load(f)
                            N_CLASSES = len(label_encoder.classes_)
                            st.toast("Reloaded latest model & artifacts.")
                    except Exception as e:
                        st.error(f"Retraining error: {e}")
