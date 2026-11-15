from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry

CSV_NAME = "renewable_energy_share_2000_2025.csv"
START, END = 2000, 2023
TOTAL = "renewables_share_elec"
SOURCES = {"solar_share_elec": "Solar", "wind_share_elec": "Wind", "hydro_share_elec": "Hydro"}
COLORS = {"Solar": "#A55000", "Wind": "#007C75", "Hydro": "#3B5FAA"}
TEAL, ORANGE, GRAY = "#007C75", "#A55000", "#64748B"
SCALE = [[0, "#D6EEEA"], [1, "#00534E"]]
ISO_CODES = {country.alpha_3 for country in pycountry.countries}


def clean_data(raw):
    """Keep observed percentages and separate the reported World total."""
    required = ["country", "iso_code", "year", TOTAL, *SOURCES]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    clean = raw[required].drop_duplicates().copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    for column in [TOTAL, *SOURCES]:
        values = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = values.where(values.between(0, 100))
    clean = clean.loc[clean["year"].between(START, END) & clean[TOTAL].notna()].copy()
    if clean.empty or (clean["year"] % 1 != 0).any():
        raise ValueError("The CSV needs valid integer years and renewable shares for 2000–2023.")
    clean["year"] = clean["year"].astype(int)
    countries = clean.loc[clean["iso_code"].isin(ISO_CODES)].copy()
    world = clean.loc[clean["country"].eq("World")].sort_values("year")
    if countries.empty or world.empty:
        raise ValueError("The CSV needs ISO-coded observations and the World aggregate.")
    if countries.duplicated(["iso_code", "year"]).any() or world["year"].duplicated().any():
        raise ValueError("Review duplicate observations for the same entity and year.")
    return clean, countries, world


def style(fig, title, height=425):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=20)),
        template="plotly_white", height=height,
        font=dict(family="Arial", size=13, color="#18283A"),
        margin=dict(t=65, r=32, b=52, l=65),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_color="#18283A"),
    )
    return fig


def world_trend(world, year):
    fig = px.line(world, x="year", y=TOTAL, markers=True,
                  labels={"year": "Year", TOTAL: "Share of global generation (%)"})
    fig.update_traces(line=dict(color=TEAL, width=3), marker=dict(size=5),
                      hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    for y in sorted({START, year}):
        value = world.loc[world["year"].eq(y), TOTAL]
        if not value.empty:
            fig.add_annotation(x=y, y=value.iloc[0], text=f"{value.iloc[0]:.2f}%",
                               showarrow=True, ax=23 if y == START else -23, ay=-29)
    fig.update_yaxes(range=[0, max(40, world[TOTAL].max() + 8)], ticksuffix="%")
    fig.update_xaxes(dtick=5)
    return style(fig, "1 · Global renewable electricity share")



def global_sources_trend(world):
    data = world[["year", *SOURCES]].copy()

    data = data.melt(
        id_vars="year",
        var_name="Renewable Source",
        value_name="Share (%)"
    )

    data["Renewable Source"] = data["Renewable Source"].map(SOURCES)

    fig = px.line(
        data,
        x="year",
        y="Share (%)",
        color="Renewable Source",
        markers=True,
        color_discrete_map=COLORS,
        labels={
            "year": "Year",
            "Share (%)": "Share of Global Electricity Generation (%)"
        }
    )

    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(range=[0, 20], ticksuffix="%", dtick=5)
    fig.update_xaxes(dtick=5)

    return style(
        fig,
        "Global Solar, Wind, and Hydro Electricity Trends (2000–2023)",
        450
    )

def world_map(snapshot, year):
    fig = px.choropleth(snapshot, locations="iso_code", locationmode="ISO-3",
                        hover_name="country", color=TOTAL, range_color=[0, 100],
                        color_continuous_scale=SCALE, labels={TOTAL: "Share (%)"},
                        hover_data={"iso_code": False, TOTAL: ":.2f"})
    fig.update_geos(projection_type="natural earth", showframe=False, showland=True,
                    landcolor="#B9C0C9", showcountries=True, countrycolor="white",
                    showcoastlines=False, lataxis_range=[-60, 85])
    fig.update_coloraxes(colorbar=dict(ticksuffix="%"))
    return style(fig, f"2 · Renewable electricity share by location, {year}", 460)


def changes(countries):
    endpoints = countries[countries["year"].isin([START, END])]
    table = endpoints.pivot(index="iso_code", columns="year", values=TOTAL)
    table = table.reindex(columns=[START, END]).dropna().reset_index()
    names = endpoints[endpoints["year"].eq(END)].set_index("iso_code")["country"]
    table["country"] = table["iso_code"].map(names)
    table["Change (pp)"] = table[END] - table[START]
    return table


def dumbbell(table, direction):
    sign = 1 if direction == "Largest increases" else -1
    selected = table[sign * table["Change (pp)"] > 0].copy()
    selected = selected.sort_values(["Change (pp)", "country"],
                                    ascending=[sign == -1, True]).head(10)
    fig = go.Figure()
    line_x, line_y = [], []
    for _, row in selected.iterrows():
        line_x.extend([row[START], row[END], None])
        line_y.extend([row["country"], row["country"], None])
    fig.add_scatter(x=line_x, y=line_y, mode="lines", line_color="#A5AEBB",
                    showlegend=False, hoverinfo="skip")
    for year, color, symbol in [(START, ORANGE, "diamond"), (END, TEAL, "circle")]:
        fig.add_scatter(x=selected[year], y=selected["country"], name=str(year),
                        mode="markers", marker=dict(color=color, size=10, symbol=symbol),
                        customdata=selected["Change (pp)"],
                        hovertemplate="%{y}: %{x:.2f}%<br>Change: %{customdata:+.2f} pp")
    # Reserve a column on the right for exact differences.
    for _, row in selected.iterrows():
        fig.add_annotation(x=1.02, xref="paper", y=row["country"],
                           text=f"{row['Change (pp)']:+.2f} pp", showarrow=False, xanchor="left")
    if selected.empty:
        fig.add_annotation(text="No matched observations in this category.", showarrow=False)
    style(fig, f"3 · {direction} in renewable share, {START}–{END}", 485)
    fig.update_xaxes(range=[-2, 102], tickvals=[0, 25, 50, 75, 100],
                     ticksuffix="%", title="Share of electricity generation (%)")
    fig.update_yaxes(categoryorder="array", categoryarray=selected["country"].tolist(),
                     autorange="reversed", automargin=True)
    fig.update_layout(margin_r=85, legend=dict(orientation="h", y=1.10))
    return fig


def distribution(countries, years):
    fig = go.Figure()
    for year in years:
        values = countries.loc[countries["year"].eq(year), TOTAL]
        if values.empty:
            continue
        label = f"{year}<br>n={len(values)}"
        fig.add_trace(go.Violin(x=[label] * len(values), y=values, name=str(year),
                                box_visible=True, points=False, spanmode="hard",
                                line_color=TEAL, fillcolor="#B7DCD6", showlegend=False))
        fig.add_annotation(x=label, y=values.median(), text=f"{values.median():.2f}%",
                           ax=39, ay=0, bgcolor="white", arrowcolor=GRAY)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", title="Renewable electricity share")
    return style(fig, "4 · Distribution across countries and territories")


def solar_wind(snapshot, year):
    points = snapshot.dropna(subset=["solar_share_elec", "wind_share_elec"])
    r = points["solar_share_elec"].corr(points["wind_share_elec"]) if len(points) >= 2 else float("nan")
    fig = px.scatter(points, x="solar_share_elec", y="wind_share_elec",
                     color=TOTAL, hover_name="country", range_color=[0, 100],
                     color_continuous_scale=SCALE,
                     labels={"solar_share_elec": "Solar share of generation (%)",
                             "wind_share_elec": "Wind share of generation (%)", TOTAL: "Renewables (%)"})
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(color="#384D56", width=0.6)))
    if points.empty:
        fig.add_annotation(text="No observations have both source shares.", showarrow=False)
    fig.update_xaxes(rangemode="tozero", ticksuffix="%")
    fig.update_yaxes(rangemode="tozero", ticksuffix="%")
    return style(fig, f"5 · Solar and wind electricity shares, {year}"), points, r


def mix_data(snapshot):
    """Normalize the three observed source shares to a common 100% total."""
    mix = snapshot.dropna(subset=list(SOURCES)).copy()
    three_source_total = mix[list(SOURCES)].sum(axis=1)
    mix = mix.loc[three_source_total.gt(0)].copy()
    long = mix.melt(id_vars=["country"], value_vars=list(SOURCES),
                    var_name="Source", value_name="Share of total electricity (%)")
    long["Source"] = long["Source"].map(SOURCES)
    value = "Share of total electricity (%)"
    long["Share of three sources (%)"] = long[value] / long.groupby("country")[value].transform("sum") * 100
    return long


def example_countries(mix):
    eligible = mix["country"].drop_duplicates().tolist()
    preferred = ["Bhutan", "Denmark", "Namibia", "Portugal"]
    matched = [country for country in preferred if country in eligible]
    return matched or sorted(eligible)[:4]


def stacked_mix(mix, selected, year):
    data = mix[mix["country"].isin(selected)]
    fig = px.bar(data, x="Share of three sources (%)", y="country", color="Source",
                 orientation="h", color_discrete_map=COLORS,
                 category_orders={"Source": list(COLORS), "country": selected},
                 hover_data={"Share of total electricity (%)": ":.2f",
                             "Share of three sources (%)": ":.2f"})
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.10))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title=None, autorange="reversed", automargin=True)
    return style(fig, f"6 · Composition of solar, wind and hydro, {year}", max(340, 185 + 48 * len(selected)))


def before_treemap(mix, selected, year):
    # Recreate the original encoding for the same countries used in the redesign.
    values = "Share of total electricity (%)"
    data = mix[mix["country"].isin(selected) & mix[values].gt(0)]
    fig = px.treemap(data, path=["country", "Source"], values=values,
                     color="Source", color_discrete_map=COLORS,
                     custom_data=["Share of three sources (%)"])
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[0]:.1f}%",
        hovertemplate="%{label}<br>Share of all generation: %{value:.2f}%"
                      "<br>Share within three sources: %{customdata[0]:.2f}%<extra></extra>",
    )
    return style(fig, f"Before · Treemap, {year}", 370)


def distribution_summary(countries, years):
    values = countries[countries["year"].isin(years)].groupby("year")[TOTAL]
    summary = values.agg(Observations="size", Median="median")
    summary["IQR (pp)"] = values.quantile(0.75) - values.quantile(0.25)
    return summary


def fmt(value, suffix="%"):
    return "Unavailable" if pd.isna(value) else f"{value:.2f}{suffix}"


def world_share(world, year):
    return world.set_index("year")[TOTAL].get(year, float("nan"))


import streamlit as st

st.set_page_config(page_title="Global Renewable Electricity", layout="wide")
st.title("Global Renewable Electricity")
st.markdown("### 2000–2023")
st.write("Explore how the global share changed, how progress varied across countries and territories, "
         "and how three renewable technologies contribute to electricity generation.")

uploaded = st.sidebar.file_uploader("Upload the project CSV", type="csv")
default_path = Path(__file__).resolve().with_name(CSV_NAME)
if uploaded is None and not default_path.is_file():
    st.info(f"Upload {CSV_NAME} or place it next to dashboard.py.")
    st.stop()

try:
    _, countries, world = clean_data(pd.read_csv(uploaded if uploaded is not None else default_path))
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

years = sorted(countries["year"].unique().tolist())
year = st.sidebar.selectbox("Snapshot year", years, index=len(years) - 1)
snapshot = countries[countries["year"].eq(year)]
paired = changes(countries)
mix = mix_data(snapshot)
world_value = world_share(world, year)
base = world_share(world, START)
cards = st.columns(4)
cards[0].metric("World share", fmt(world_value))
cards[1].metric("Country and territory median", fmt(snapshot[TOTAL].median()))
cards[2].metric("Countries and territories", len(snapshot))
cards[3].metric(f"World change since {START}", fmt(world_value - base, " pp"))

overview, change_tab, sources_tab, redesign_tab = st.tabs(
    ["Global overview", "Country change", "Electricity sources", "Before and after"]
)

with overview:
    st.plotly_chart(world_trend(world, year), width="stretch")
    st.caption("The global series uses the World aggregate reported in the source. "
               "The second annotation marks the snapshot year.")
    st.plotly_chart(global_sources_trend(world), width="stretch")
    st.caption("Solar and wind shares increased over time, while hydropower remained the largest of the three sources.")

    st.plotly_chart(world_map(snapshot, year), width="stretch")
    st.caption(f"{len(snapshot)} territories and countries have an observation in {year}. "
               "Gray shows areas without a rendered observation. Light teal starts at 0%. "
               "Some small territories are easier to locate in the table.")
    with st.expander("Explore the values behind the map"):
        st.dataframe(snapshot[["country", TOTAL]].rename(columns={
            "country": "Country or territory", TOTAL: "Renewable share (%)"
        }).sort_values("Country or territory"), hide_index=True)

with change_tab:
    direction = st.radio("Rank changes", ["Largest increases", "Largest decreases"], horizontal=True)
    st.plotly_chart(dumbbell(paired, direction), width="stretch")
    delta = paired["Change (pp)"]
    st.caption(f"{len(paired)} entities have data in both {START} and {END}. "
               f"{delta.gt(0).sum()} increased, {delta.lt(0).sum()} decreased "
               f"and {delta.eq(0).sum()} stayed the same. The difference is measured in percentage points.")
    selected_years = st.multiselect("Years for distribution comparison", years,
                                   default=[y for y in [2000, 2010, 2020, 2023] if y in years])
    if selected_years:
        st.plotly_chart(distribution(countries, sorted(selected_years)), width="stretch")
        st.dataframe(distribution_summary(countries, selected_years).round(2))
        st.caption("IQR is the spread of the middle 50%. The set of economies with observations can differ by year.")
    else:
        st.info("Select at least one year to see the distribution.")

with sources_tab:
    fig, points, r = solar_wind(snapshot, year)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(points)} complete observations. Pearson r = {fmt(r, '')}. "
               "Both axes show a share of total electricity generation.")
    options = sorted(mix["country"].unique().tolist())
    selection = st.multiselect("Countries and territories to compare", options,
                               default=example_countries(mix), max_selections=8)
    if selection:
        st.plotly_chart(stacked_mix(mix, selection, year), width="stretch")
        st.caption("Each bar totals 100% of solar, wind and hydro combined. "
                   "The hover label also shows the share of total electricity for each source.")
    else:
        st.info("Select a country or territory with complete data and a positive three-source total.")

with redesign_tab:
    last = countries[countries["year"].eq(END)]
    comparison_mix = mix_data(last)
    examples = example_countries(comparison_mix)
    if examples:
        left, right = st.columns(2)
        left.plotly_chart(before_treemap(comparison_mix, examples, END), width="stretch")
        right.plotly_chart(stacked_mix(comparison_mix, examples, END), width="stretch", key="redesign_after")
        left.caption("Before · Rectangle area represents share of total generation. "
                     "The text inside represents share within the three sources.")
        right.caption("After · Bar length uses the same denominator as the composition percentage. "
                      "Equal-length rows make countries easier to compare.")
    else:
        st.info("No 2023 observations are suitable for the redesign comparison.")

st.caption("Source: Kaggle, Renewable Energy EDA input CSV. Analysis window: 2000–2023.")
