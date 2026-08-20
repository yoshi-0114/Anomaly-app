from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "public_data" / "learning_events.csv.gz"

ACCENT = "#3568d4"
ACCENT_LIGHT = "#dce8ff"
NORMAL_COLOR = "#aeb8c5"
ANOMALY_COLOR = "#d9485f"
TEXT_COLOR = "#233044"

FEATURE_COLUMNS = [
    "act_count",
    "unique_count",
    "active_minutes",
    "pages_viewed",
    "navigation_ratio",
    "backtrack_ratio",
    "annotation_ratio",
    "comprehension_ratio",
    "search_ratio",
]

FEATURE_LABELS = {
    "act_count": "操作数",
    "unique_count": "操作種類数",
    "active_minutes": "活動時間",
    "pages_viewed": "閲覧ページ数",
    "navigation_ratio": "ページ移動率",
    "backtrack_ratio": "戻る操作率",
    "annotation_ratio": "書き込み率",
    "comprehension_ratio": "理解度操作率",
    "search_ratio": "検索・ジャンプ率",
}

OPERATION_LABELS = {
    "OPEN": "資料を開く",
    "CLOSE": "資料を閉じる",
    "NEXT": "次ページ",
    "PREV": "前ページ",
    "PAGE_JUMP": "ページジャンプ",
    "ADD_MARKER": "マーカー追加",
    "DELETE_MARKER": "マーカー削除",
    "ADD_BOOKMARK": "ブックマーク追加",
    "DELETE_BOOKMARK": "ブックマーク削除",
    "BOOKMARK_JUMP": "ブックマークへ移動",
    "ADD_MEMO": "メモ追加",
    "CHANGE_MEMO": "メモ変更",
    "DELETE_MEMO": "メモ削除",
    "MEMO_TEXT_CHANGE_HISTORY": "メモ編集",
    "ADD_HW_MEMO": "手書きメモ追加",
    "CLEAR_HW_MEMO": "手書きメモ削除",
    "GETIT": "理解した",
    "NOTGETIT": "理解できない",
    "SEARCH": "検索",
    "SEARCH_JUMP": "検索結果へ移動",
    "LINK_CLICK": "リンクを開く",
    "TIMER_STOP": "タイマー停止",
}

OPERATION_CATEGORIES = {
    "ページ移動": {"NEXT", "PREV", "PAGE_JUMP"},
    "資料の開閉": {"OPEN", "CLOSE"},
    "マーカー": {"ADD_MARKER", "DELETE_MARKER"},
    "ブックマーク": {"ADD_BOOKMARK", "DELETE_BOOKMARK", "BOOKMARK_JUMP"},
    "メモ": {"ADD_MEMO", "CHANGE_MEMO", "DELETE_MEMO", "MEMO_TEXT_CHANGE_HISTORY", "ADD_HW_MEMO", "CLEAR_HW_MEMO"},
    "理解度": {"GETIT", "NOTGETIT"},
    "検索・リンク": {"SEARCH", "SEARCH_JUMP", "LINK_CLICK"},
    "その他": {"TIMER_STOP"},
}


st.set_page_config(
    page_title="学習行動ダッシュボード｜公開デモ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def configure_japanese_font() -> None:
    candidates = [
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def add_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.18); }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stMetricLabel"] { font-weight: 600; }
        .public-badge {
            display: inline-block;
            color: #2756b3;
            background: #e7efff;
            border: 1px solid #c8d9ff;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-bottom: 0.35rem;
        }
        .section-note {
            border-left: 4px solid #3568d4;
            background: rgba(53, 104, 212, 0.07);
            border-radius: 0 8px 8px 0;
            padding: 0.7rem 0.9rem;
            margin: 0.2rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_public_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(
        path,
        compression="gzip",
        parse_dates=["eventtime"],
        dtype={
            "userid": "string",
            "course_id": "category",
            "operationname": "category",
            "contentsid": "string",
            "devicecode": "category",
            "grade": "category",
        },
    )
    events["marker"] = events["marker"].fillna("")

    work = events.assign(
        is_navigation=events["operationname"].isin(["NEXT", "PREV", "PAGE_JUMP"]).astype(float),
        is_backtrack=(events["operationname"] == "PREV").astype(float),
        is_annotation=events["operationname"].isin(
            [
                "ADD_MARKER",
                "DELETE_MARKER",
                "ADD_MEMO",
                "CHANGE_MEMO",
                "DELETE_MEMO",
                "MEMO_TEXT_CHANGE_HISTORY",
                "ADD_HW_MEMO",
                "CLEAR_HW_MEMO",
            ]
        ).astype(float),
        is_comprehension=events["operationname"].isin(["GETIT", "NOTGETIT"]).astype(float),
        is_search=events["operationname"].isin(
            ["PAGE_JUMP", "BOOKMARK_JUMP", "SEARCH", "SEARCH_JUMP", "LINK_CLICK"]
        ).astype(float),
    )
    group_keys = ["userid", "course_id", "week", "grade"]
    weekly = (
        work.groupby(group_keys, observed=True)
        .agg(
            act_count=("operationname", "size"),
            unique_count=("operationname", "nunique"),
            first_event=("eventtime", "min"),
            last_event=("eventtime", "max"),
            pages_viewed=("pageno", "nunique"),
            navigation_ratio=("is_navigation", "mean"),
            backtrack_ratio=("is_backtrack", "mean"),
            annotation_ratio=("is_annotation", "mean"),
            comprehension_ratio=("is_comprehension", "mean"),
            search_ratio=("is_search", "mean"),
            injected_anomaly=("is_injected_anomaly", "max"),
        )
        .reset_index()
    )
    weekly["active_minutes"] = (
        (weekly["last_event"] - weekly["first_event"]).dt.total_seconds() / 60
    ).clip(lower=0.1)
    return events, weekly


@st.cache_data(show_spinner=False)
def analyze_weekly(weekly: pd.DataFrame, contamination: float) -> pd.DataFrame:
    result = weekly.copy().reset_index(drop=True)
    matrix = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    detector = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    prediction = detector.fit_predict(scaled)
    raw_score = -detector.score_samples(scaled)
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)

    result["is_anomaly"] = prediction == -1
    result["raw_anomaly_score"] = raw_score
    result["attention_score"] = pd.Series(raw_score).rank(method="average", pct=True).mul(100).to_numpy()
    result["pca_x"] = projection[:, 0]
    result["pca_y"] = projection[:, 1]
    return result


@st.cache_data(show_spinner=False)
def assign_behavior_types(result: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    typed = result.copy()
    matrix = typed[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    typed["behavior_type"] = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    ).fit_predict(scaled)
    typed["behavior_type"] = typed["behavior_type"].map(lambda value: f"タイプ {int(value) + 1}")
    return typed


def plot_anomaly_map(result: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    normal = result[~result["is_anomaly"]]
    anomaly = result[result["is_anomaly"]]
    ax.scatter(
        normal["pca_x"],
        normal["pca_y"],
        s=22,
        color=NORMAL_COLOR,
        alpha=0.48,
        label="通常範囲",
    )
    scatter = ax.scatter(
        anomaly["pca_x"],
        anomaly["pca_y"],
        s=42,
        c=anomaly["attention_score"],
        cmap="Reds",
        alpha=0.9,
        edgecolor="#a42f43",
        linewidth=0.35,
        label="要確認",
    )
    if not anomaly.empty:
        fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03, label="要確認スコア")
    ax.set_title("学習行動の分布", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("行動の特徴軸 1")
    ax.set_ylabel("行動の特徴軸 2")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_weekly_trend(filtered: pd.DataFrame):
    trend = (
        filtered.groupby("week")
        .agg(operations=("act_count", "sum"), students=("userid", "nunique"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(10.5, 3.8))
    ax1.bar(trend["week"], trend["operations"], color=ACCENT, alpha=0.82, width=0.65, label="操作数")
    ax1.set_xlabel("週")
    ax1.set_ylabel("操作数")
    ax1.set_xticks(trend["week"], [f"{week}週" for week in trend["week"]])
    ax1.grid(axis="y", alpha=0.16)
    ax2 = ax1.twinx()
    ax2.plot(
        trend["week"],
        trend["students"],
        color=ANOMALY_COLOR,
        marker="o",
        linewidth=2.2,
        label="参加学生数",
    )
    ax2.set_ylabel("参加学生数")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax1.set_title("週ごとのデータ量", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_operation_categories(detail_events: pd.DataFrame):
    operation_to_category = {
        operation: category
        for category, operations in OPERATION_CATEGORIES.items()
        for operation in operations
    }
    counts = (
        detail_events["operationname"]
        .astype("string")
        .map(operation_to_category)
        .fillna("その他")
        .value_counts()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(counts.index, counts.values, color=ACCENT, alpha=0.84)
    ax.set_xlabel("操作数")
    ax.set_title("操作カテゴリの内訳", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.16)
    for index, value in enumerate(counts.values):
        ax.text(value, index, f" {value:,}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_behavior_map(typed: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    type_names = sorted(typed["behavior_type"].unique(), key=lambda value: int(value.split()[-1]))
    palette = sns.color_palette("tab10", n_colors=len(type_names))
    for name, color in zip(type_names, palette):
        group = typed[typed["behavior_type"] == name]
        ax.scatter(group["pca_x"], group["pca_y"], s=24, alpha=0.68, label=name, color=color)
    ax.set_title("行動タイプの分布", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("行動の特徴軸 1")
    ax.set_ylabel("行動の特徴軸 2")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_behavior_profile(typed: pd.DataFrame):
    profile = typed.groupby("behavior_type")[FEATURE_COLUMNS].mean()
    standardized = (profile - typed[FEATURE_COLUMNS].mean()) / typed[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)
    standardized = standardized.rename(columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    sns.heatmap(
        standardized,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "全体平均との差"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("タイプ別の特徴", loc="left", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def format_ranking(result: pd.DataFrame) -> pd.DataFrame:
    ranking = result.sort_values("attention_score", ascending=False).copy()
    ranking["判定"] = np.where(ranking["is_anomaly"], "要確認", "通常範囲")
    ranking["対象"] = ranking["userid"].astype(str)
    ranking["コース"] = ranking["course_id"].astype(str)
    ranking["週"] = ranking["week"].map(lambda value: f"{int(value)}週目")
    ranking["成績"] = ranking["grade"].astype(str)
    ranking["要確認スコア"] = ranking["attention_score"].round(1)
    ranking["操作数"] = ranking["act_count"].astype(int)
    ranking["操作種類数"] = ranking["unique_count"].astype(int)
    ranking["閲覧ページ数"] = ranking["pages_viewed"].astype(int)
    return ranking[
        ["判定", "対象", "コース", "週", "成績", "要確認スコア", "操作数", "操作種類数", "閲覧ページ数"]
    ]


configure_japanese_font()
add_page_styles()

st.title("学習行動 異常検知ダッシュボード")
st.caption("授業中の操作ログから、普段と異なる学習行動と行動タイプを確認する公開デモです。")

if not DATA_PATH.exists():
    st.error("公開用ダミーデータがありません。`python3 scripts/generate_public_dummy_data.py` を実行してください。")
    st.stop()

with st.spinner("公開用ダミーデータを読み込んでいます…"):
    events, weekly_features = load_public_data(str(DATA_PATH))

all_courses = sorted(events["course_id"].astype(str).unique())
all_weeks = sorted(int(value) for value in events["week"].unique())

with st.sidebar:
    st.header("分析条件")
    selected_courses = st.multiselect(
        "コース",
        all_courses,
        default=all_courses,
        help="すべて架空のコースです。",
    )
    selected_weeks = st.multiselect(
        "表示する週",
        all_weeks,
        default=all_weeks,
        format_func=lambda value: f"{value}週目",
    )
    contamination_percent = st.slider(
        "要確認とする割合",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        format="%d%%",
        help="行動パターンが全体から離れている学生・週を、指定割合だけ抽出します。",
    )
    st.caption("初期状態では1〜8週目をすべて分析します。")
    st.divider()
    st.subheader("公開データ")
    st.write(f"学生数　**{events['userid'].nunique():,}人**")
    st.write(f"期間　**{min(all_weeks)}〜{max(all_weeks)}週目**")
    st.write(f"操作ログ　**{len(events):,}件**")
    st.success("実在する学生・授業の情報は含んでいません。")

if not selected_courses or not selected_weeks:
    st.warning("コースと週を1つ以上選択してください。")
    st.stop()

filtered_weekly = weekly_features[
    weekly_features["course_id"].astype(str).isin(selected_courses)
    & weekly_features["week"].isin(selected_weeks)
].copy()
filtered_events = events[
    events["course_id"].astype(str).isin(selected_courses)
    & events["week"].isin(selected_weeks)
].copy()

if len(filtered_weekly) < 10:
    st.warning("分析対象が少なすぎます。コースまたは週を追加してください。")
    st.stop()

with st.spinner("行動パターンを分析しています…"):
    result = analyze_weekly(filtered_weekly, contamination_percent / 100)

anomaly_result = result[result["is_anomaly"]].sort_values("attention_score", ascending=False)

st.info(
    "この画面の結果は合成データによるデモです。『要確認』は相対的に珍しい行動を示し、学生の評価や診断を意味しません。"
)

metric_columns = st.columns(4)
metric_columns[0].metric("対象学生", f"{result['userid'].nunique():,}人")
metric_columns[1].metric("対象の学生・週", f"{len(result):,}件")
metric_columns[2].metric("操作ログ", f"{int(result['act_count'].sum()):,}件")
metric_columns[3].metric("要確認", f"{len(anomaly_result):,}件", f"{len(anomaly_result) / len(result) * 100:.1f}%")

overview_tab, ranking_tab, detail_tab, behavior_tab = st.tabs(
    ["概要", "要確認ランキング", "学生・週の詳細", "行動タイプ"]
)

with overview_tab:
    st.markdown(
        '<div class="section-note">灰色は通常範囲、赤は全体から離れた行動です。点は「1人 × 1週」を表します。</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        anomaly_figure = plot_anomaly_map(result)
        st.pyplot(anomaly_figure, width="stretch")
        plt.close(anomaly_figure)
    with right:
        st.subheader("判定サマリー")
        summary = pd.DataFrame(
            [
                {
                    "判定": "通常範囲",
                    "件数": int((~result["is_anomaly"]).sum()),
                    "割合": f"{(~result['is_anomaly']).mean() * 100:.1f}%",
                    "平均操作数": f"{result.loc[~result['is_anomaly'], 'act_count'].mean():.1f}",
                },
                {
                    "判定": "要確認",
                    "件数": int(result["is_anomaly"].sum()),
                    "割合": f"{result['is_anomaly'].mean() * 100:.1f}%",
                    "平均操作数": f"{result.loc[result['is_anomaly'], 'act_count'].mean():.1f}",
                },
            ]
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.markdown("##### 読み方")
        st.write(
            "1. 全体分布で外れた点を確認  \n"
            "2. ランキングで対象を絞る  \n"
            "3. 個別の操作履歴で背景を確認"
        )
        st.caption("要確認スコアは、選択中のデータ内での相対順位（0〜100）です。")

    weekly_figure = plot_weekly_trend(result)
    st.pyplot(weekly_figure, width="stretch")
    plt.close(weekly_figure)

with ranking_tab:
    st.subheader("要確認ランキング")
    st.caption("スコアが高い順に表示しています。スコアは異常の確率ではありません。")
    ranking = format_ranking(anomaly_result)
    st.dataframe(
        ranking,
        hide_index=True,
        width="stretch",
        height=470,
        column_config={
            "要確認スコア": st.column_config.ProgressColumn(
                "要確認スコア",
                min_value=0,
                max_value=100,
                format="%.1f",
            )
        },
    )
    st.download_button(
        "ランキングをCSVでダウンロード",
        data=ranking.to_csv(index=False).encode("utf-8-sig"),
        file_name="synthetic_anomaly_ranking.csv",
        mime="text/csv",
        width="content",
    )

with detail_tab:
    st.subheader("学生・週の詳細")
    st.caption("ランキングで気になった対象の、実際の操作内訳と時系列を確認します。")
    default_user = str(anomaly_result.iloc[0]["userid"]) if not anomaly_result.empty else str(result.iloc[0]["userid"])
    user_options = sorted(result["userid"].astype(str).unique())
    selected_user = st.selectbox(
        "対象学生",
        user_options,
        index=user_options.index(default_user) if default_user in user_options else 0,
    )
    user_rows = result[result["userid"].astype(str) == selected_user]
    user_weeks = sorted(int(value) for value in user_rows["week"].unique())
    default_week = int(user_rows.sort_values("attention_score", ascending=False).iloc[0]["week"])
    selected_detail_week = st.selectbox(
        "対象週",
        user_weeks,
        index=user_weeks.index(default_week),
        format_func=lambda value: f"{value}週目",
    )
    detail = user_rows[user_rows["week"] == selected_detail_week].iloc[0]
    detail_events = filtered_events[
        (filtered_events["userid"].astype(str) == selected_user)
        & (filtered_events["week"] == selected_detail_week)
    ].sort_values("eventtime")

    detail_metrics = st.columns(5)
    detail_metrics[0].metric("判定", "要確認" if detail["is_anomaly"] else "通常範囲")
    detail_metrics[1].metric("要確認スコア", f"{detail['attention_score']:.1f}")
    detail_metrics[2].metric("操作数", f"{int(detail['act_count']):,}")
    detail_metrics[3].metric("操作種類数", f"{int(detail['unique_count'])}")
    detail_metrics[4].metric("閲覧ページ数", f"{int(detail['pages_viewed'])}")

    chart_column, log_column = st.columns([0.8, 1.2], gap="large")
    with chart_column:
        category_figure = plot_operation_categories(detail_events)
        st.pyplot(category_figure, width="stretch")
        plt.close(category_figure)
    with log_column:
        st.markdown("##### 操作履歴")
        display_events = detail_events[
            ["eventtime", "operationname", "contentsid", "pageno", "marker", "memo_length", "devicecode"]
        ].copy()
        display_events["eventtime"] = display_events["eventtime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_events["operationname"] = (
            display_events["operationname"].astype("string").map(OPERATION_LABELS).fillna(display_events["operationname"].astype("string"))
        )
        display_events.columns = ["操作時刻", "操作", "教材ID", "ページ", "マーカー", "メモ文字数", "端末"]
        st.dataframe(display_events, hide_index=True, width="stretch", height=390)

with behavior_tab:
    st.subheader("行動タイプ")
    st.caption("似た特徴を持つ学生・週をまとめます。タイプ番号に優劣の意味はありません。")
    max_clusters = min(8, max(3, len(result) // 10))
    cluster_count = st.slider("行動タイプ数", min_value=3, max_value=max_clusters, value=min(5, max_clusters))
    typed_result = assign_behavior_types(result, cluster_count)
    type_left, type_right = st.columns([0.9, 1.1], gap="large")
    with type_left:
        behavior_map = plot_behavior_map(typed_result)
        st.pyplot(behavior_map, width="stretch")
        plt.close(behavior_map)
    with type_right:
        behavior_profile = plot_behavior_profile(typed_result)
        st.pyplot(behavior_profile, width="stretch")
        plt.close(behavior_profile)

    type_summary = (
        typed_result.groupby("behavior_type")
        .agg(
            学生週数=("userid", "size"),
            学生数=("userid", "nunique"),
            平均操作数=("act_count", "mean"),
            要確認件数=("is_anomaly", "sum"),
        )
        .reset_index()
        .rename(columns={"behavior_type": "行動タイプ"})
    )
    type_summary["平均操作数"] = type_summary["平均操作数"].round(1)
    st.dataframe(type_summary, hide_index=True, width="stretch")

st.divider()
