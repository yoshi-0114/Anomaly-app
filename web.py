import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import OpenLA as la
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.ensemble import IsolationForest
import gc
from sklearn.cluster import KMeans
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

st.set_page_config(page_title="学習行動 異常検知ダッシュボード", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

@st.cache_data
def load_data(min_week, max_week, courses):
    df = pd.DataFrame()
    user_grades = {}
    for course in courses:
        course_info, event_stream = la.start_analysis(files_dir="./Students", course_id=course)
        users = sorted(course_info.user_id(), key=lambda x: int(x.split("_U")[1]))
        for u in users:
            try:
                user_grades[u] = course_info.user_grade(u)
            except:
                user_grades[u] = None
        for week in range(min_week, max_week + 1):
            try:
                event_stream_during = la.select_by_lecture_time(course_info, event_stream, lecture_week=week, timing="during")
            except:
                continue
            sorted_df = event_stream_during.df.sort_values(
                by=["userid", "eventtime"],
                key=lambda x: x.map(lambda y: int(re.search(r'U(\d+)', y).group(1))) if x.name == "userid" else x
            )
            sorted_df["week"] = week
            df = pd.concat([df, sorted_df], ignore_index=True)
            # df = df[~df["operationname"].isin(["NEXT", "PREV"])]

    actions = sorted(pd.concat([df])["operationname"].unique())
    action2id = {a: i+2 for i, a in enumerate(actions)}
    action2id["[PAD]"] = 0
    action2id["[MASK]"] = 1
    id2action = {v:k for k,v in action2id.items()}

    return df, action2id, id2action, user_grades

class ActionBERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, max_position_embeddings=512):
        super().__init__()
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_position_embeddings, d_model)
        self.dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, action_ids, attention_mask=None, src_key_padding_mask=None):
        B, L = action_ids.shape
        action_emb = self.action_embedding(action_ids)
        pos = torch.arange(L, device=action_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)

        x = self.dropout(action_emb + pos_emb)

        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(out)

        return logits
    
def BERT_structure(model_type):
    if model_type == 'tiny':
        d_model, n_heads, n_layers = 128, 2, 2
        batch_size = 32
    elif model_type == 'mini':
        d_model, n_heads, n_layers = 256, 4, 4
        batch_size = 32
    elif model_type == 'small':
        d_model, n_heads, n_layers = 512, 8, 4
        batch_size = 32
    elif model_type == 'medium':
        d_model, n_heads, n_layers = 512, 8, 8
        batch_size = 32
    elif model_type == 'base':
        d_model, n_heads, n_layers = 768, 12, 12
        batch_size = 32
    elif model_type == 'large':
        d_model, n_heads, n_layers = 1024, 16, 24
        batch_size = 16
    
    return d_model, n_heads, n_layers, batch_size

@st.cache_resource
def load_model(model_path, model_type, max_seq_len, action2id):
    d_model, n_heads, n_layers, _ = BERT_structure(model_type)
    model = ActionBERT(vocab_size=len(action2id), d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_position_embeddings=max_seq_len).to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def detect_anomalies(model, df, action2id, id2action, seq_len, avg_mode, target_grades, method, contamination):
    model.eval()
    
    week_stats = df.groupby(["userid", "week"])["operationname"].agg(
        act_count="count", 
        unique_count=lambda x: x.nunique(),
        operations=lambda x: list(x)
    ).reset_index().rename(columns={"userid": "user"})
    
    student_stats = df.groupby("userid").agg(
        act_count=("operationname", "count"), 
        unique_count=("operationname", lambda x: x.nunique()),
        operations=("operationname", lambda x: list(x)),
        min_week=("week", "min"),
        max_week=("week", "max")
    ).reset_index().rename(columns={"userid": "user"})

    all_sequences = []
    window_user = []
    window_meta = []
    user_window_action = {}

    for (sid, week), group in df.groupby(["userid", "week"], sort=False):
        group = group.sort_values("eventtime")
        actions = list(group["operationname"])
        weeks = list(group["week"]) if "week" in group.columns else [0] * len(actions)
        
        actions_ids = [action2id[a] for a in actions]

        L = len(actions_ids)

        count = 0
        for t in range(0, L):
            start = t
            end = seq_len + t
            
            if t == 0 or t % seq_len == 0:
                window_seq = actions_ids[start:end]
                
                if len(window_seq) != seq_len and count > 0:
                    start = L - seq_len
                    end = L
                    window_seq = actions_ids[start:end]
                
                current_week = weeks[start] if start < len(weeks) else weeks[-1]
                window_act_count = len(window_seq)
                window_unique_count = len(set(window_seq))
                
                window_meta.append({
                    "user": sid,
                    "week": current_week,
                    "window_idx": count,
                    "act_count": window_act_count,
                    "unique_count": window_unique_count
                })

                user_window_action[(sid, current_week, count)] = [id2action[a] for a in window_seq]

                count += 1
                pad_len = seq_len - len(window_seq)
                input_seq = [action2id["[PAD]"]] * pad_len + window_seq

                all_sequences.append(input_seq)
                window_user.append(sid)

    batch_size = 8
    input_ids_tensor = torch.tensor(all_sequences, device=device)
    emb_list = []

    with torch.no_grad():
        for i in range(0, len(input_ids_tensor), batch_size):
            batch = input_ids_tensor[i:i+batch_size]
            action_emb = model.action_embedding(batch)
            pos_emb = model.pos_embed(torch.arange(seq_len, device=device).unsqueeze(0))
            # x = model.dropout(action_emb + pos_emb)
            # src_key_padding_mask = batch == action2id["[PAD]"]
            # emb = model.encoder(x, src_key_padding_mask=src_key_padding_mask)
            encoder = pos = True
            if not encoder:
                emb = action_emb + pos_emb if pos else action_emb
            else:
                x = model.dropout(action_emb + pos_emb if pos else action_emb)
                src_key_padding_mask = batch == action2id["[PAD]"]
                emb = model.encoder(x, src_key_padding_mask=src_key_padding_mask)
            emb_list.append(emb.cpu())
    emb = torch.cat(emb_list, dim=0)

    embeddings_array = emb.cpu().numpy().reshape(-1, emb.size(-1))
    labels_array = input_ids_tensor.cpu().numpy().reshape(-1)

    split_embeddings = embeddings_array.reshape(-1, seq_len, embeddings_array.shape[1])
    split_labels = labels_array.reshape(-1, seq_len)

    split_embeddings_mean = []
    for i in range(split_labels.shape[0]):
        labels = split_labels[i]
        valid = (labels != action2id["[PAD]"]) & (labels != action2id["[MASK]"])
        split_embeddings_mean.append(split_embeddings[i][valid].mean(axis=0))
    split_embeddings_mean = np.array(split_embeddings_mean)

    window_df = pd.DataFrame(window_meta)
    window_df["emb"] = list(split_embeddings_mean)
    user_window_action_df = pd.DataFrame([{"user": k[0], "week": k[1], "window_idx": k[2], "operations": v} for k, v in user_window_action.items()])

    if avg_mode == "window":
        X = split_embeddings_mean
        meta_df = window_df[["user", "week", "window_idx", "act_count", "unique_count"]]
        meta_df = pd.merge(meta_df, user_window_action_df, on=["user", "week", "window_idx"], how="left")
    elif avg_mode == "window_week":
        grouped = window_df.groupby(["user", "week"])
        X = np.stack(grouped["emb"].apply(lambda x: np.mean(np.stack(x), axis=0)))
        meta_df = grouped.size().reset_index()[["user", "week"]]
        meta_df = pd.merge(meta_df, week_stats, on=["user", "week"], how="left")
        st.write("window_week")
    elif avg_mode == "window_week_student":
        week_df = (window_df.groupby(["user", "week"])["emb"].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index())
        grouped_user = week_df.groupby("user")
        X = np.stack(grouped_user["emb"].apply(lambda x: np.mean(np.stack(x), axis=0)))
        meta_df = grouped_user.size().reset_index()[["user"]]
        meta_df = pd.merge(meta_df, student_stats, on="user", how="left")
    elif avg_mode == "window_student":
        grouped = window_df.groupby("user")
        X = np.stack(grouped["emb"].apply(lambda x: np.mean(np.stack(x), axis=0)))
        meta_df = grouped.size().reset_index()[["user"]]
        meta_df = pd.merge(meta_df, student_stats, on="user", how="left")

    meta_df["grade"] = meta_df["user"].map(target_grades)

    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    anomaly_preds = iso_forest.fit_predict(X)
    anomaly_scores = -iso_forest.score_samples(X)
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)

    emb_2d = reducer.fit_transform(X)
    
    meta_df["operations"] = meta_df["operations"].apply(
        lambda ops: [
            op.replace("NOTGETIT", "NOT GET IT")
            .replace("GETIT", "GET IT")
            .replace("_", " ")
            for op in ops
        ] if isinstance(ops, list) else ops
    )

    return emb_2d, anomaly_preds, anomaly_scores, meta_df, X

def create_summary_row(df, label, total_len):
    count = len(df)
    ratio = (count / total_len) * 100 if total_len > 0 else 0
    
    act_mean = pd.to_numeric(df['act_count'], errors='coerce').mean()
    if pd.isna(act_mean):
        act_mean = 0.0

    grades = ['A', 'B', 'C', 'D', 'F']
    grade_counts = df['grade'].value_counts()
    valid_grade_total = df['grade'].isin(grades).sum()
    
    grade_ratios = {}
    for g in grades:
        g_count = grade_counts.get(g, 0)
        grade_ratios[g] = (g_count / valid_grade_total * 100) if valid_grade_total > 0 else 0
        
    return {
        "判定結果": label,
        "データ数": count,
        "割合 [%]": round(ratio, 1),
        "平均操作回数": round(act_mean, 1),
        "A": round(grade_ratios['A'], 1),
        "B": round(grade_ratios['B'], 1),
        "C": round(grade_ratios['C'], 1),
        "D": round(grade_ratios['D'], 1),
        "F": round(grade_ratios['F'], 1)
    }

st.title("学習行動 異常検知ダッシュボード")

min_week = 1
max_week = 8
epoch = 300
model_path = f"./model_{min_week}-{max_week}.pth"
all_course = [
    "A-2019", "A-2020", "A-2021", "A-2022",
    "B-2019", "B-2020",
    "C-2021-1", "C-2021-2", "C-2022-1",
    "D-2020", "D-2021", "D-2022",
    "E-2020-1", "E-2020-2", "E-2021",
    "F-2021", "G-2021"]

with st.spinner('データを読み込んでいます...'):
    all_df, action2id, id2action, user_grades = load_data(min_week, max_week, all_course)
if all_df.empty: st.stop()

with st.spinner('モデルを読み込んでいます...'):
    max_seq_len = 512
    model_type = "base"
    model = load_model(model_path, model_type, max_seq_len, action2id)
if model is None: st.stop()

target_course = [c for c in all_course if re.search(r'2022', c)]
target_df, _, _, target_grades = load_data(min_week, max_week, target_course)

st.sidebar.header("設定")

with st.sidebar.form("analysis_settings"):
    available_weeks = sorted(target_df["week"].unique())
    selected_weeks = st.multiselect(
        "表示する週を選択",
        options=available_weeks,
        default=[available_weeks[0]] if available_weeks else []
    )
    
    avg_mode_options = {
        "ウィンドウ単位": "window",
        "週単位": "window_week",
        # "学生単位 - 週平均": "window_week_student",
        "学生単位": "window_student"
    }

    selected_mode_label = st.selectbox("分析単位", list(avg_mode_options.keys()))
    avg_mode = avg_mode_options[selected_mode_label]

    contamination_rate = st.slider("異常判定の閾値", 0.01, 0.20, 0.05, 0.01)

    run_analysis = st.form_submit_button("実行")

if run_analysis:
    if not selected_weeks:
        st.warning("表示する週を選択してください")
        st.stop()

    st.session_state["reset_counter"] += 1

    keys_to_reset = ["detail_user", "detail_week", "compare_course", "compare_selections", "n_clusters_slider"]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]

    target_select_df = target_df[target_df["week"].isin(selected_weeks)]

    st.write(
        f"対象学生数：{target_select_df['userid'].nunique()}人 / "
        f"操作ログ数：{len(target_select_df):,} / モード：**{selected_mode_label}**"
    )

    method = "tSNE"

    with st.spinner('データを解析中...'):
        emb_2d, anomaly_preds, anomaly_scores, meta_df, X = detect_anomalies(
            model,
            target_select_df,
            action2id,
            id2action,
            max_seq_len,
            avg_mode,
            target_grades,
            method,
            contamination_rate
        )

        st.session_state["anomaly_results"] = {
            "emb_2d": emb_2d,
            "anomaly_preds": anomaly_preds,
            "anomaly_scores": anomaly_scores,
            "meta_df": meta_df,
            "X": X,
            "target_select_df": target_select_df
        }

if "anomaly_results" in st.session_state:
    results = st.session_state["anomaly_results"]
    emb_2d = results["emb_2d"]
    anomaly_preds = results["anomaly_preds"]
    anomaly_scores = results["anomaly_scores"]
    meta_df = results["meta_df"]
    X = results["X"]
    target_select_df = results["target_select_df"]
    if emb_2d is not None:
        st.subheader("異常検知結果")
        
        normal_df = meta_df[anomaly_preds == 1]
        anomaly_df = meta_df[anomaly_preds == -1]
        total_len = len(meta_df)
        
        normal_stats = create_summary_row(normal_df, "正常", total_len)
        anomaly_stats = create_summary_row(anomaly_df, "異常", total_len)
        
        # columns = pd.MultiIndex.from_tuples([
        #     ("判定結果", ""), ("データ数", ""), ("割合 [%]", ""), ("平均操作回数", ""),
        #     ("成績分布 [%]", "A"), ("成績分布 [%]", "B"), ("成績分布 [%]", "C"), ("成績分布 [%]", "D"), ("成績分布 [%]", "F")
        # ])
        # summary_data = [
        #     [normal_stats["判定結果"], normal_stats["データ数"], normal_stats["割合 [%]"], normal_stats["平均操作回数"], 
        #      normal_stats["A"], normal_stats["B"], normal_stats["C"], normal_stats["D"], normal_stats["F"]],
        #     [anomaly_stats["判定結果"], anomaly_stats["データ数"], anomaly_stats["割合 [%]"], anomaly_stats["平均操作回数"], 
        #      anomaly_stats["A"], anomaly_stats["B"], anomaly_stats["C"], anomaly_stats["D"], anomaly_stats["F"]]
        # ]
        # summary_df = pd.DataFrame(summary_data, columns=columns)
        # st.dataframe(summary_df.style.format(precision=1), use_container_width=True)

        summary_html = f"""
        <table style="width:100%; border-collapse: collapse; text-align: center; margin-bottom: 20px;" border="1">
            <thead style="background-color: #f0f2f6;">
                <tr>
                    <th rowspan="2" style="padding: 8px;">判定結果</th>
                    <th rowspan="2" style="padding: 8px;">データ数</th>
                    <th rowspan="2" style="padding: 8px;">割合 [%]</th>
                    <th rowspan="2" style="padding: 8px;">平均操作回数</th>
                    <th colspan="5" style="padding: 8px;">成績分布 [%]</th>
                </tr>
                <tr>
                    <th style="padding: 8px;">A</th>
                    <th style="padding: 8px;">B</th>
                    <th style="padding: 8px;">C</th>
                    <th style="padding: 8px;">D</th>
                    <th style="padding: 8px;">F</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 8px;">{normal_stats['判定結果']}</td>
                    <td style="padding: 8px;">{normal_stats['データ数']}</td>
                    <td style="padding: 8px;">{normal_stats['割合 [%]']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['平均操作回数']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['A']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['B']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['C']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['D']:.1f}</td>
                    <td style="padding: 8px;">{normal_stats['F']:.1f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">{anomaly_stats['判定結果']}</td>
                    <td style="padding: 8px;">{anomaly_stats['データ数']}</td>
                    <td style="padding: 8px;">{anomaly_stats['割合 [%]']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['平均操作回数']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['A']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['B']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['C']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['D']:.1f}</td>
                    <td style="padding: 8px;">{anomaly_stats['F']:.1f}</td>
                </tr>
            </tbody>
        </table>
        """
        
        st.markdown(summary_html, unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2 = st.columns([0.8, 1.2])

        with col1:
            st.subheader(f"行動ベクトルの可視化")
            fig, ax = plt.subplots(figsize=(5.55, 5))
            normal_mask = anomaly_preds == 1
            anomaly_mask = anomaly_preds == -1
            
            ax.scatter(emb_2d[normal_mask, 0], emb_2d[normal_mask, 1], c='lightgrey', label='Normal', s=20, alpha=0.5)
            if any(anomaly_mask):
                scatter = ax.scatter(emb_2d[anomaly_mask, 0], emb_2d[anomaly_mask, 1], 
                                     c=anomaly_scores[anomaly_mask], cmap='Reds', label='Anomaly', s=30, alpha=0.9)
                # fig.colorbar(scatter, ax=ax)
                
            ax.legend(loc='upper left')
            st.pyplot(fig)
            
        with col2:
            st.subheader("異常度ランキング")

            results = []
            for idx, row in meta_df.iterrows():
                res_dict = {
                    "学生": row["user"],
                    "成績": row["grade"],
                    "異常度": anomaly_scores[idx],
                    "状態": "異常" if anomaly_preds[idx] == -1 else "正常",
                }
                if avg_mode == "window":
                    res_dict["週"] = row["week"]
                    res_dict["Window"] = row["window_idx"]
                elif avg_mode == "window_week":
                    res_dict["週"] = row["week"]
                else:
                    res_dict["対象週"] = f"{row['min_week']} - {row['max_week']}"
                
                res_dict["操作数"] = row["act_count"]
                res_dict["操作種類数"] = row["unique_count"]
                # res_dict["主な操作"] = ", ".join(pd.Series(row["operations"]).value_counts().head(3).index.tolist())
                if "operations" in row and isinstance(row["operations"], (list, pd.Series)):
                    res_dict["主な操作"] = ", ".join(pd.Series(row["operations"]).value_counts().head(3).index.tolist())
                else:
                    res_dict["主な操作"] = ""
                results.append(res_dict)
                
            df_results = pd.DataFrame(results)
            if df_results.empty or "状態" not in df_results.columns:
                st.info("表示可能な異常度結果がありませんでした")
            else:
                df_anomaly = df_results.query("状態 == '異常'").sort_values("異常度", ascending=False).reset_index(drop=True)
                df_anomaly = df_anomaly.drop(columns=["状態"])
                st.write(f"検出された異常データ数：{len(df_anomaly)}")
                st.dataframe(df_anomaly.style.format({"異常度": "{:.4f}"}), hide_index=True, use_container_width=True)

        torch.cuda.empty_cache()
        gc.collect()

        st.markdown("---")
        st.subheader("学生ごとの操作ログ")

        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            user_list = sorted(
                target_select_df["userid"].unique(),
                key=lambda x: (
                    x.split("_")[0],
                    int(x.split("_")[1][1:])
                )
            )

            user_key = f"detail_user_{st.session_state['reset_counter']}"
            
            # selected_user = st.selectbox("詳細を表示するユーザーIDを選択", options=user_list, index=0)
            selected_user = st.selectbox(
                "詳細を表示するユーザーIDを選択",
                options=user_list,
                format_func=lambda x: f"{x} ({target_grades.get(x, 'N/A')})",
                key=user_key
            )

        with col_select2:
            user_weeks = sorted(target_select_df[target_select_df["userid"] == selected_user]["week"].unique())
            # user_weeks = [num for num in range(min(selected_weeks), max(selected_weeks) + 1)]
            if user_weeks:
                week_key = f"detail_week_{st.session_state['reset_counter']}"
                selected_detail_week = st.selectbox(
                    "詳細を表示する週を選択",
                    options=user_weeks,
                    key=week_key)
            else:
                selected_detail_week = None
                st.info("このユーザーのデータはありません")

        if selected_detail_week is not None:
            user_log_df = target_select_df[
                (target_select_df["userid"] == selected_user) & 
                (target_select_df["week"] == selected_detail_week)
            ].sort_values("eventtime")

            st.write(f"**{selected_user}** の **{selected_detail_week}週目** の操作履歴（計 {len(user_log_df)} 操作）")

            display_df = user_log_df[["operationname", "contentsid", "pageno", "marker", "memo_length", "eventtime"]].reset_index(drop=True)
            display_df["operationname"] = (
                display_df["operationname"]
                .str.replace("NOTGETIT", "NOT GET IT", regex=False)
                .str.replace("GETIT", "GET IT", regex=False)
                .str.replace("_", " ", regex=False)
            )
            display_df.columns = ["操作名", "コンテンツID", "ページ番号", "マーカー", "メモの長さ", "操作時刻"]
            st.dataframe(display_df, use_container_width=True)

        st.markdown("---")

        st.subheader("クラスタリング設定")
        current_key = f"n_clusters_slider_{st.session_state['reset_counter']}"
        n_clusters = st.slider("クラスタ数を選択", min_value=2, max_value=20, value=14, step=1, key=current_key)
        # n_clusters = st.slider("クラスタ数を選択", min_value=2, max_value=20, value=14, step=1, key="n_clusters_slider")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        meta_df["Cluster"] = cluster_labels
        
        col_cluster1, col_cluster2 = st.columns([0.8, 1.2])
        col_cluster1_analysis, col_cluster2_analysis = st.columns([0.9, 1.1])

        with col_cluster1:
            st.markdown("**行動ベクトルの可視化 (クラスタリング)**")
            fig_cluster, ax_cluster = plt.subplots(figsize=(5.55, 5))
            cmap = plt.colormaps.get_cmap('tab20')
            scatter_cluster = ax_cluster.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap=cmap, s=20, alpha=0.7)
            
            legend = ax_cluster.legend(*scatter_cluster.legend_elements(), title="Clusters", loc="upper left", bbox_to_anchor=(1.05, 1))
            ax_cluster.add_artist(legend)
            fig_cluster.tight_layout()
            st.pyplot(fig_cluster)
            
        with col_cluster2:
            st.markdown("**各クラスタの成績分布**")
            
            grade_order = ['A', 'B', 'C', 'D', 'F', "None"]
            grade_counts = pd.crosstab(meta_df['Cluster'], meta_df['grade'])
            
            valid_grades = [g for g in grade_order if g in grade_counts.columns]
            
            if valid_grades:
                grade_counts = grade_counts[valid_grades]
                fig_grade, ax_grade = plt.subplots(figsize=(8, 5))
                
                grade_counts.plot(kind='bar', stacked=True, ax=ax_grade, colormap='RdYlBu_r')
                
                ax_grade.set_ylabel("Number of Students", fontsize=9)
                ax_grade.set_xlabel("Cluster ID", fontsize=9)
                ax_grade.set_xticklabels(ax_grade.get_xticklabels(), rotation=0)
                ax_grade.legend(loc='upper left', fontsize=9)
                fig_grade.tight_layout()
                
                st.pyplot(fig_grade, use_container_width=True)
            else:
                st.info("表示可能な成績データがありません")

        with col_cluster1_analysis:
            st.markdown("**各クラスタの操作数**")
            
            fig_box, ax_box = plt.subplots(figsize=(5.55, 5))
            sns.boxplot(data=meta_df, x="Cluster", y="act_count", hue="Cluster", palette="Set3", legend=False, ax=ax_box)
            # ax_box.set_title("Distribution of Operation Counts per Cluster")
            ax_box.set_xlabel("Cluster ID", fontsize=9)
            ax_box.set_ylabel("Number of Operations", fontsize=9)
            ax_box.grid(axis='y', linestyle='--', alpha=0.7)
            fig_box.tight_layout()
            st.pyplot(fig_box)
            
        with col_cluster2_analysis:
            st.markdown("**操作比率**")

            exploded_df = meta_df.explode("operations")
            action_counts = pd.crosstab(exploded_df["Cluster"], exploded_df["operations"])
            action_ratios = action_counts.div(action_counts.sum(axis=1), axis=0)
            
            fig_heat, ax_heat = plt.subplots(figsize=(5.55, 5))
            sns.heatmap(action_ratios, cmap="YlGnBu", annot=False, ax=ax_heat)
            # ax_heat.set_title("Action Distribution Heatmap (Row Normalized)")
            ax_heat.set_ylabel("Cluster ID", fontsize=9)
            # ax_heat.set_xlabel("Operation Name")
            ax_heat.set_xlabel("", fontsize=0)
            ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=8)
            fig_heat.tight_layout()
            st.pyplot(fig_heat)

        st.markdown("---")
        st.subheader("学生別 操作カテゴリ集計")
        
        categorys = {
            "ページ開閉": ["OPEN", "CLOSE"],
            "ページ遷移": ["NEXT", "PREV", "PAGE JUMP"],
            "マーカー": ["ADD MARKER", "DELETE MARKER"],
            "ブックマーク": ["ADD BOOKMARK", "DELETE BOOKMARK", "BOOKMARK JUMP"],
            "メモ": ["ADD MEMO", "DELETE MEMO", "CHANGE MEMO", "MEMO TEXT CHANGE HISTORY", "MEMO JUMP"],
            "手書きメモ": ["ADD HW MEMO", "CLEAR HW MEMO", "UNDO HW MEMO", "REDO HW MEMO"],
            "理解度": ["GET IT", "NOT GET IT"],
            "検索": ["SEARCH", "SEARCH JUMP"],
            "リンク": ["LINK CLICK"],
            "リコメンド": ["OPEN RECOMMENDATION", "CLOSE RECOMMENDATION", "ADD RECOMMENDATION", "DELETE RECOMMENDATION", "CLICK RECOMMENDATION", "REGIST CONTENTS"],
            "タイマー": ["TIMER STOP", "TIMER PAUSE"]
        }

        op2cat = {op: cat for cat, ops in categorys.items() for op in ops}

        if "compare_selections" not in st.session_state:
            st.session_state["compare_selections"] = []

        available_courses = sorted(list(set([u.split("_")[0] for u in meta_df["user"].unique()])))
        col_course, col_user = st.columns([0.3, 0.7])
        with col_course:
            course_key = f"compare_course_{st.session_state['reset_counter']}"
            selected_course = st.selectbox(
                "対象のコースを選択",
                options=available_courses,
                key=course_key
            )

        course_users = sorted(
            [u for u in meta_df["user"].unique() if u.startswith(selected_course)],
            key=lambda x: int(x.split("_")[1][1:])
        )
        
        user_week_options = []
        for user in course_users:
            if "week" in meta_df.columns:
                user_weeks = sorted(meta_df[meta_df["user"] == user]["week"].unique())
                for w in user_weeks:
                    user_week_options.append(f"{user} ({w})")
            user_week_options.append(f"{user} (全週)")

        for sel in st.session_state["compare_selections"]:
            if sel not in user_week_options:
                user_week_options.append(sel)

        with col_user:
            selected_combinations = st.multiselect(
                "表示したい「学生と週」の組み合わせを選択",
                options=user_week_options,
                key="compare_selections"
            )

        if selected_combinations:
            comparison_data = []
            
            for selection in selected_combinations:
                u_name, w_str = selection.split(" (")
                w_str = w_str.rstrip(")")
                
                if w_str == "全週":
                    temp_df = meta_df[meta_df["user"] == u_name].copy()
                else:
                    w_num = int(w_str.replace("週目", ""))
                    temp_df = meta_df[(meta_df["user"] == u_name) & (meta_df["week"] == w_num)].copy()
                
                if not temp_df.empty:
                    exploded = temp_df.explode("operations")
                    exploded["category"] = exploded["operations"].map(op2cat).fillna("その他")
                    
                    cat_counts = exploded["category"].value_counts().to_dict()
                    cat_counts["学生 (週)"] = selection
                    comparison_data.append(cat_counts)
            
            if not comparison_data:
                st.warning("選択した条件に一致するデータがありません。")
            else:
                user_category_df = pd.DataFrame(comparison_data).fillna(0)
                
                category_order = list(categorys.keys()) + ["その他"]
                valid_cols = [c for c in category_order if c in user_category_df.columns]
                user_category_df = user_category_df[['学生 (週)'] + valid_cols]
                
                user_category_df["計"] = user_category_df.drop(columns="学生 (週)").sum(axis=1)

                st.markdown("**操作カテゴリ集計表**")
                st.dataframe(user_category_df, hide_index=True, use_container_width=True)
    else:
        st.warning("解析可能なデータがありませんでした")
