import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# page setup
st.set_page_config(page_title="national parks — seasonality & overnight use", layout="wide")
st.title("national parks — seasonality & overnight use")
st.caption("data: nps public use statistics (2000 – last calendar year)")

# load data
data_path = Path("data/nps_visitation.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    st.caption(f"loaded data from: `{data_path.name}`")
else:
    st.error(
        "csv not found in `/data`. "
        "place the file `Query Builder for Public Use Statistics (2000- Last Calendar Year).csv` into the `data/` folder."
    )
    st.stop()

# clean and engineer
# coerce numeric columns if present
num_cols = [
    "RecreationVisits","NonRecreationVisits","RecreationHours","NonRecreationHours",
    "ConcessionerLodging","ConcessionerCamping","TentCampers","RVCampers",
    "Backcountry","NonRecreationOvernightStays","MiscellaneousOvernightStays",
]
for c in num_cols:
    if c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"": np.nan, "NA": np.nan})
            .astype(float)
        )

# check required columns
needed = ["ParkName","UnitCode","Region","State","Year","Month","RecreationVisits","RecreationHours"]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"missing required columns: {missing}")
    st.stop()

# convert year and month
df["Year"]  = pd.to_numeric(df["Year"], errors="coerce")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
df = df.dropna(subset=["Year","Month","RecreationVisits","RecreationHours"]).copy()

# create datetime from year and month
df["Date"] = pd.to_datetime(
    dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1),
    errors="coerce"
)

# normalize region and state, build state tokens (handles "CA, NV" etc.)
df["Region"] = df["Region"].astype(str).str.strip()
df["State"] = (
    df["State"].astype(str)
    .str.upper()
    .str.replace(r"[^A-Z,; ]", "", regex=True)
    .str.replace(";", ",")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df["StateTokens"] = df["State"].apply(
    lambda s: [t.strip() for t in s.split(",") if t.strip()] if isinstance(s, str) and s.strip() else []
)

# create overnight metrics
overnight_parts = [c for c in [
    "ConcessionerLodging","ConcessionerCamping","TentCampers","RVCampers",
    "Backcountry","NonRecreationOvernightStays","MiscellaneousOvernightStays"
] if c in df.columns]

df["Overnight"] = df[overnight_parts].sum(axis=1, min_count=1) if overnight_parts else np.nan
df["OvernightShare"] = np.where(df["RecreationVisits"] > 0, df["Overnight"] / df["RecreationVisits"], np.nan)
df["HoursPerVisit"]  = np.where(df["RecreationVisits"] > 0, df["RecreationHours"] / df["RecreationVisits"], np.nan)

# sidebar filters
parks      = sorted(df["ParkName"].dropna().unique().tolist())
regions    = sorted(df["Region"].dropna().unique().tolist())
all_states = sorted({t for toks in df["StateTokens"] for t in toks})
yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())

st.sidebar.header("filters")
park_sel  = st.sidebar.multiselect("park(s)",   parks,   default=parks[:3] if parks else [])
region_sel= st.sidebar.multiselect("region(s)", regions, default=regions if regions else [])
state_sel = st.sidebar.multiselect("state(s)",  all_states)
yr_range  = st.sidebar.slider("year range", yr_min, yr_max, (max(yr_min, yr_max - 10), yr_max))

# apply filters
if state_sel:
    state_mask = df["StateTokens"].apply(lambda toks: any(t in state_sel for t in toks))
else:
    state_mask = True

mask = (
    (df["ParkName"].isin(park_sel) if park_sel else True) &
    (df["Region"].isin(region_sel) if region_sel else True) &
    (df["Year"].between(yr_range[0], yr_range[1])) &
    (state_mask)
)
view = df[mask].copy()
if view.empty:
    st.warning("no data after filters. adjust selections and try again.")
    st.stop()

# kpis
c1, c2, c3, c4 = st.columns(4)
c1.metric("total visits (filtered)", f"{int(view['RecreationVisits'].sum()):,}")
c2.metric("median overnight share", f"{(view['OvernightShare'].median(skipna=True) * 100):.1f}%")
c3.metric("median hours / visit", f"{view['HoursPerVisit'].median(skipna=True):.2f}")
c4.metric("distinct parks (filtered)", view["ParkName"].nunique())

# optional state summary table when state filter is used
if state_sel:
    st.caption("visits by selected state(s)")
    state_sum = (
        view.explode("StateTokens")
            .query("StateTokens in @state_sel")
            .groupby("StateTokens", as_index=False)["RecreationVisits"]
            .sum()
            .sort_values("RecreationVisits", ascending=False)
    )
    st.dataframe(state_sum.rename(columns={"StateTokens":"state","RecreationVisits":"visits"}), use_container_width=True)

# list distinct parks for the current filter
st.subheader("parks included (filtered)")
parks_df = (
    view[["ParkName","UnitCode","Region","State"]]
    .drop_duplicates()
    .sort_values(["ParkName","UnitCode"])
    .reset_index(drop=True)
)
st.dataframe(parks_df.rename(columns={
    "ParkName":"park name", "UnitCode":"unit code", "Region":"region", "State":"state"
}), use_container_width=True)

st.divider()

# trend
st.subheader("monthly recreation visits — trend")
trend = view.groupby("Date", as_index=False)["RecreationVisits"].sum()
st.line_chart(trend.set_index("Date"))

# seasonality
st.subheader("seasonality by region (average monthly recreation visits)")
heat = (
    view.groupby(["Region", "Month"])["RecreationVisits"]
        .mean()
        .reset_index()
        .pivot(index="Month", columns="Region", values="RecreationVisits")
        .fillna(0)
)
st.dataframe(heat.style.format("{:,.0f}"), use_container_width=True)

# camping mix (keeps zero-na parks; adds search + top-n)
camp_cols_needed = {"TentCampers","RVCampers","Backcountry"}
if camp_cols_needed.issubset(view.columns):
    st.subheader("camping mix — % by park (tent vs rv vs backcountry)")
    mix = (view.groupby("ParkName")[["TentCampers","RVCampers","Backcountry"]]
              .sum(min_count=1)
              .fillna(0))
    totals = mix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mix_share = (mix.div(totals.replace(0, np.nan), axis=0) * 100)
    mix_share = mix_share.fillna(0).round(1)

    top_n = st.slider("rows to show", 10, 200, 50)
    search_txt = st.text_input("filter by park name (optional)")
    sort_col = st.selectbox("sort by", ["TentCampers","RVCampers","Backcountry"], index=0)

    tbl = mix_share.copy()
    if search_txt:
        tbl = tbl[tbl.index.str.contains(search_txt, case=False, na=False)]
    st.dataframe(tbl.sort_values(sort_col, ascending=False).head(top_n), use_container_width=True)
else:
    st.info("camping mix columns not all present; skipping mix table.")

# park finder (quick debug to locate a specific park like glacier np)
with st.expander("park finder"):
    q = st.text_input("search park", "Glacier NP")
    hits = view[view["ParkName"].str.contains(q, case=False, na=False)]
    st.write("matches:", sorted(hits["ParkName"].unique()))
    by_park = (hits.groupby("ParkName")[["RecreationVisits","TentCampers","RVCampers","Backcountry"]]
                   .sum(min_count=1).fillna(0))
    st.dataframe(by_park, use_container_width=True)

st.divider()

# stats
st.subheader("stats")

# anova test on july visits by region
july = view[view["Month"] == 7].dropna(subset=["RecreationVisits"])
if july["Region"].nunique() >= 2:
    groups = [g["RecreationVisits"].values for _, g in july.groupby("Region")]
    if all(len(g) >= 3 for g in groups):
        f, p = stats.f_oneway(*groups)
        p_fmt = f"{p:.3e}" if p < 1e-4 else f"{p:.4f}"
        st.write(f"anova (july recreationvisits by region): f = {f:.2f}, p = {p_fmt}")
    else:
        st.write("anova: not enough observations in every region group (need ≥3 per region).")
else:
    st.write("anova: not enough distinct regions in current filter.")

# regression of hours/visit on overnight share
reg = view.dropna(subset=["HoursPerVisit", "OvernightShare"])[["HoursPerVisit", "OvernightShare"]]
if len(reg) >= 30 and reg["OvernightShare"].var() > 0:
    slope, intercept, r, p, se = stats.linregress(reg["OvernightShare"].values, reg["HoursPerVisit"].values)
    p_fmt = f"{p:.3e}" if p < 1e-4 else f"{p:.4f}"
    st.write(f"regression (hours/visit ~ overnight share): slope = {slope:.2f}, r = {r:.3f}, r² = {r**2:.3f}, p = {p_fmt}")
else:
    st.write("regression: not enough rows or variation in overnight share.")

st.divider()

# additional methods
st.subheader("additional methods")

# a) correlation heatmap (pearson)
with st.expander("correlation heatmap"):
    numeric = view[["RecreationVisits","Overnight","OvernightShare","RecreationHours","HoursPerVisit"]].copy()
    numeric = numeric.dropna(how="any")
    if len(numeric) >= 20:
        corr = numeric.corr(method="pearson")
        fig, ax = plt.subplots()
        im = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center")
        st.pyplot(fig)
    else:
        st.write("not enough rows for a stable correlation matrix (need ≥20).")

# b) kmeans clustering with pca scatter
with st.expander("kmeans clustering (park usage profiles)"):
    agg = view.groupby(["ParkName","UnitCode"], as_index=False).agg(
        total_visits=("RecreationVisits","sum"),
        avg_overnight_share=("OvernightShare","mean"),
        avg_hours_per_visit=("HoursPerVisit","mean")
    ).dropna()
    if len(agg) >= 5:
        k = st.slider("clusters (k)", 2, 6, 3)
        scaler = StandardScaler()
        X = scaler.fit_transform(agg[["total_visits","avg_overnight_share","avg_hours_per_visit"]])
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        agg["cluster"] = labels

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        fig, ax = plt.subplots()
        sc = ax.scatter(X2[:,0], X2[:,1], c=labels)
        for i in range(len(agg)):
            ax.annotate(str(agg["cluster"].iloc[i]), (X2[i,0], X2[i,1]))
        ax.set_xlabel("pca1"); ax.set_ylabel("pca2")
        st.pyplot(fig)

        st.dataframe(
            agg.sort_values(["cluster","total_visits"], ascending=[True, False]).reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.write("not enough parks in the current filter for clustering (need ≥5).")

# c) predictive models (linear regression + neural network)
with st.expander("predictive models (linear regression + neural network)"):
    tmp = view.dropna(subset=["RecreationVisits"]).copy()
    if len(tmp) >= 60:
        # cyclical month encoding
        tmp["month_sin"] = np.sin(2*np.pi*tmp["Month"]/12.0)
        tmp["month_cos"] = np.cos(2*np.pi*tmp["Month"]/12.0)

        # choose a park to plot
        park_for_plot = st.selectbox(
            "select a park to plot actual vs predicted",
            sorted(tmp["ParkName"].unique())
        )
        tmp = tmp.sort_values("Date")

        # train/test split by time
        cutoff_year = st.slider(
            "train/test cutoff year",
            int(tmp["Year"].min())+3,
            int(tmp["Year"].max())-1,
            int(min(2018, tmp["Year"].max()-1))
        )
        train = tmp[tmp["Year"] <= cutoff_year]
        test  = tmp[tmp["Year"] >  cutoff_year]

        # numeric features
        X_cols_num = ["Year","month_sin","month_cos"]
        X_train_num = train[X_cols_num]
        X_test_num  = test[X_cols_num]

        # one-hot for region (handle old/new sklearn api)
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        enc.fit(train[["Region"]])
        X_train = np.hstack([X_train_num.values, enc.transform(train[["Region"]])])
        X_test  = np.hstack([X_test_num.values,  enc.transform(test[["Region"]])])
        y_train = train["RecreationVisits"].values
        y_test  = test["RecreationVisits"].values

        # linear regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # neural network (mlp regressor)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                           random_state=42, max_iter=1000)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)

        # rmse compatible with all sklearn versions
        rmse_lr  = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
        rmse_mlp = float(np.sqrt(mean_squared_error(y_test, y_pred_mlp)))
        st.write(f"linear regression rmse: {rmse_lr:,.0f}")
        st.write(f"neural network rmse: {rmse_mlp:,.0f}")

        # actual vs predicted for selected park
        test_plot = (
            test[test["ParkName"] == park_for_plot][["Date","RecreationVisits"]]
            .rename(columns={"RecreationVisits":"actual"})
            .copy()
        )
        if not test_plot.empty:
            test_dates = test["Date"].reset_index(drop=True)
            pred_df = pd.DataFrame({"Date": test_dates, "pred_lr": y_pred_lr, "pred_mlp": y_pred_mlp})
            plot_df = test_plot.merge(pred_df, on="Date", how="left").sort_values("Date")

            fig, ax = plt.subplots()
            ax.plot(plot_df["Date"], plot_df["actual"], label="actual")
            ax.plot(plot_df["Date"], plot_df["pred_lr"], label="pred_lr")
            ax.plot(plot_df["Date"], plot_df["pred_mlp"], label="pred_mlp")
            ax.set_xlabel("date"); ax.set_ylabel("recreation visits")
            ax.legend()
            st.pyplot(fig)

            st.dataframe(plot_df.tail(12).reset_index(drop=True), use_container_width=True)
        else:
            st.write("no test-period rows for the selected park after the cutoff year.")
    else:
        st.write("not enough rows for predictive modeling (need ≥60).")

st.caption(
    "notes: small p-values are formatted in scientific notation when very small. "
    "anova compares mean july visits across regions. in the regression, r is pearson correlation between hours/visit and overnight share."
)
