import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="爱库存经营自动分析工具 v1", layout="wide")


# =========================
# 固定业务口径
# =========================
BASE_COMMISSION_RATE = 0.18
TECH_SERVICE_RATE = 0.12
TRANSACTION_FEE_RATE = 0.006
FLOW_RADAR_RATE = 0.003
SHIPPING_INSURANCE_PER_AD = 0.6
FREIGHT_PER_AD = 5.0

GIFT_COST_MAP = {
    "维C赠品": 5.0,
    "运动水杯": 4.25,
    "小熊行李箱": 25.0,
}


# =========================
# 通用工具
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_excel_any(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    return clean_columns(df)


def as_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"缺少字段，候选字段：{candidates}")
    return None


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class ColumnConfig:
    platform: Optional[str]
    store_name: str
    ad_no: str
    order_status: str
    aftersale_type: Optional[str]
    pay_time: Optional[str]
    sku_code: Optional[str]
    barcode: Optional[str]
    product_name: str
    spec_text: Optional[str]
    qty: str
    paid_amount: str
    marketing_return: str
    activity_name: Optional[str]


# =========================
# 字段识别
# =========================
def detect_order_columns(df: pd.DataFrame) -> ColumnConfig:
    return ColumnConfig(
        platform=pick_col(df, ["平台"], required=False),
        store_name=pick_col(df, ["店铺名称"]),
        ad_no=pick_col(df, ["ad单号", "AD单号", "ad单", "母单号"]),
        order_status=pick_col(df, ["订单状态"]),
        aftersale_type=pick_col(df, ["售后类型"], required=False),
        pay_time=pick_col(df, ["支付时间", "付款时间", "支付成功时间"], required=False),
        sku_code=pick_col(df, ["货号"], required=False),
        barcode=pick_col(df, ["条形码"], required=False),
        product_name=pick_col(df, ["商品名称", "产品名称"]),
        spec_text=pick_col(df, ["规格"], required=False),
        qty=pick_col(df, ["数量", "商品数量", "购买数量"]),
        paid_amount=pick_col(df, ["实付金额", "支付金额", "买家实付"]),
        marketing_return=pick_col(df, ["营销后回款价"]),
        activity_name=pick_col(df, ["活动名称"], required=False),
    )


def validate_mapping_columns(link_df: pd.DataFrame, sales_df: pd.DataFrame, product_df: pd.DataFrame) -> None:
    for col in ["平台", "店铺名称", "货号", "销售规格ID"]:
        if col not in link_df.columns:
            raise KeyError(f"店铺链接映射表缺少字段：{col}")

    for col in ["销售规格ID", "标准产品ID", "产品总成本"]:
        if col not in sales_df.columns:
            raise KeyError(f"销售规格映射表缺少字段：{col}")

    for col in ["标准产品ID", "标准产品名称"]:
        if col not in product_df.columns:
            raise KeyError(f"标准产品主档表缺少字段：{col}")


# =========================
# 业务处理
# =========================
def mark_gift_name(name: str) -> Tuple[bool, Optional[str]]:
    text = str(name).strip()
    if "维C赠品" in text:
        return True, "维C赠品"
    if "运动水杯" in text or "水杯" in text:
        return True, "运动水杯"
    if "行李箱" in text:
        return True, "小熊行李箱"
    return False, None


def preprocess_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnConfig]:
    cfg = detect_order_columns(df)
    work = df.copy()

    # 标准化字段
    for col in [
        cfg.store_name,
        cfg.ad_no,
        cfg.order_status,
        cfg.product_name,
    ]:
        work[col] = as_str(work[col])

    if cfg.aftersale_type:
        work[cfg.aftersale_type] = as_str(work[cfg.aftersale_type])

    if cfg.platform:
        work[cfg.platform] = as_str(work[cfg.platform])
    else:
        work["平台"] = "爱库存"
        cfg.platform = "平台"

    if cfg.sku_code:
        work[cfg.sku_code] = as_str(work[cfg.sku_code])
    else:
        work["货号"] = ""
        cfg.sku_code = "货号"

    if cfg.barcode:
        work[cfg.barcode] = as_str(work[cfg.barcode])
    else:
        work["条形码"] = ""
        cfg.barcode = "条形码"

    if cfg.spec_text:
        work[cfg.spec_text] = as_str(work[cfg.spec_text])
    else:
        work["规格"] = ""
        cfg.spec_text = "规格"

    if cfg.activity_name:
        work[cfg.activity_name] = as_str(work[cfg.activity_name])
    else:
        work["活动名称"] = "未标注活动"
        cfg.activity_name = "活动名称"

    work[cfg.qty] = as_num(work[cfg.qty])
    work[cfg.paid_amount] = as_num(work[cfg.paid_amount])
    work[cfg.marketing_return] = as_num(work[cfg.marketing_return])

    # 赠品识别
    gift_flags = work[cfg.product_name].apply(mark_gift_name)
    work["是否赠品"] = gift_flags.apply(lambda x: x[0]) | (as_str(work[cfg.sku_code]) == "")
    work["赠品名称"] = gift_flags.apply(lambda x: x[1] if x[1] else "")

    # 空货号赠品：货号自动匹配为条形码
    work["货号修正"] = as_str(work[cfg.sku_code])
    empty_sku_mask = work["货号修正"].eq("") & work["是否赠品"]
    work.loc[empty_sku_mask, "货号修正"] = as_str(work.loc[empty_sku_mask, cfg.barcode])

    # 有效订单
    aftersale_series = as_str(work[cfg.aftersale_type]) if cfg.aftersale_type else pd.Series("", index=work.index)
    work["是否无效订单"] = (
        as_str(work[cfg.order_status]).eq("订单取消")
        | aftersale_series.eq("退货退款")
    )
    work["是否有效订单"] = ~work["是否无效订单"]

    # 主商品
    work["是否主商品"] = work["是否有效订单"] & (~work["是否赠品"])

    return work, cfg


def build_cost_mapping(
    link_df: pd.DataFrame, sales_df: pd.DataFrame, product_df: pd.DataFrame
) -> pd.DataFrame:
    validate_mapping_columns(link_df, sales_df, product_df)

    link = link_df.copy()
    sales = sales_df.copy()
    product = product_df.copy()

    for col in ["平台", "店铺名称", "货号", "销售规格ID"]:
        link[col] = as_str(link[col])

    for col in ["销售规格ID", "标准产品ID"]:
        sales[col] = as_str(sales[col])

    for col in ["标准产品ID", "标准产品名称"]:
        product[col] = as_str(product[col])

    if "产品总成本" in sales.columns:
        sales["产品总成本"] = as_num(sales["产品总成本"])

    merged = link.merge(
        sales[["销售规格ID", "标准产品ID", "产品总成本"]],
        on="销售规格ID",
        how="left",
    ).merge(
        product[["标准产品ID", "标准产品名称"]],
        on="标准产品ID",
        how="left",
    )

    merged = merged.rename(columns={"产品总成本": "映射产品总成本"})
    return merged


def attach_costs(
    orders: pd.DataFrame,
    cfg: ColumnConfig,
    cost_map: pd.DataFrame,
) -> pd.DataFrame:
    work = orders.copy()

    join_cols = [
        (cfg.platform, "平台"),
        (cfg.store_name, "店铺名称"),
        ("货号修正", "货号"),
    ]

    left_on = [x[0] for x in join_cols]
    right_on = [x[1] for x in join_cols]

    work = work.merge(
        cost_map[["平台", "店铺名称", "货号", "销售规格ID", "标准产品ID", "标准产品名称", "映射产品总成本"]],
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    work["产品总成本"] = work["映射产品总成本"].fillna(0.0)
    work["运费"] = 0.0

    # 运费按 ad 单号分摊到主商品
    ad_counts = (
        work[work["是否主商品"]]
        .groupby(cfg.ad_no, dropna=False)
        .size()
        .rename("主商品行数")
        .reset_index()
    )
    work = work.merge(ad_counts, on=cfg.ad_no, how="left")
    work["主商品行数"] = work["主商品行数"].fillna(0)

    main_mask = work["是否主商品"] & (work["主商品行数"] > 0)
    work.loc[main_mask, "运费"] = FREIGHT_PER_AD / work.loc[main_mask, "主商品行数"]

    return work


def calc_metrics(work: pd.DataFrame, cfg: ColumnConfig) -> Dict[str, float]:
    main = work[work["是否主商品"]].copy()
    gifts = work[work["是否赠品"]].copy()

    gmv = main[cfg.paid_amount].sum()
    qty = main[cfg.qty].sum()
    ad_cnt = main[cfg.ad_no].nunique()

    marketing_return = main[cfg.marketing_return].sum()
    gross_profit = gmv - main["产品总成本"].sum() - main["运费"].sum()

    base_commission = gmv * BASE_COMMISSION_RATE
    tech_service_fee = gmv * TECH_SERVICE_RATE
    high_commission = gmv - marketing_return - tech_service_fee - base_commission
    commission_rate = safe_div(base_commission + high_commission, gmv)

    transaction_fee = gmv * TRANSACTION_FEE_RATE
    shipping_insurance = ad_cnt * SHIPPING_INSURANCE_PER_AD
    flow_radar_fee = gmv * FLOW_RADAR_RATE
    settlement_amount = marketing_return - shipping_insurance - flow_radar_fee

    settlement_profit = settlement_amount - main["产品总成本"].sum() - main["运费"].sum()

    gift_cost = 0.0
    for gift_name, unit_cost in GIFT_COST_MAP.items():
        gift_qty = gifts.loc[gifts["赠品名称"] == gift_name, cfg.qty].sum()
        gift_cost += gift_qty * unit_cost

    return {
        "GMV": gmv,
        "主商品销量": qty,
        "有效主商品订单数": float(ad_cnt),
        "营销后回款价": marketing_return,
        "毛利": gross_profit,
        "基础佣金": base_commission,
        "平台技术服务费": tech_service_fee,
        "高佣加码": high_commission,
        "商品佣金率": commission_rate,
        "平台交易手续费": transaction_fee,
        "运费险": shipping_insurance,
        "流量雷达服务费": flow_radar_fee,
        "结算金额": settlement_amount,
        "结算后利润": settlement_profit,
        "赠品总成本": gift_cost,
    }


def build_sku_analysis(work: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    main = work[work["是否主商品"]].copy()
    if main.empty:
        return pd.DataFrame()

    grouped = main.groupby(
        ["货号修正", cfg.product_name, "标准产品名称"],
        dropna=False,
        as_index=False,
    ).agg(
        GMV=(cfg.paid_amount, "sum"),
        销量=(cfg.qty, "sum"),
        营销后回款价=(cfg.marketing_return, "sum"),
        产品总成本=("产品总成本", "sum"),
        运费=("运费", "sum"),
    )

    grouped["基础佣金"] = grouped["GMV"] * BASE_COMMISSION_RATE
    grouped["平台技术服务费"] = grouped["GMV"] * TECH_SERVICE_RATE
    grouped["高佣加码"] = (
        grouped["GMV"]
        - grouped["营销后回款价"]
        - grouped["平台技术服务费"]
        - grouped["基础佣金"]
    )
    grouped["商品佣金率"] = (grouped["基础佣金"] + grouped["高佣加码"]) / grouped["GMV"].replace(0, pd.NA)

    # ad 单级费用按货号占 GMV 比例简化分摊
    total_gmv = grouped["GMV"].sum()
    if total_gmv > 0:
        grouped["运费险"] = grouped["GMV"] / total_gmv * (main[cfg.ad_no].nunique() * SHIPPING_INSURANCE_PER_AD)
    else:
        grouped["运费险"] = 0.0
    grouped["流量雷达服务费"] = grouped["GMV"] * FLOW_RADAR_RATE
    grouped["结算金额"] = grouped["营销后回款价"] - grouped["运费险"] - grouped["流量雷达服务费"]
    grouped["毛利"] = grouped["GMV"] - grouped["产品总成本"] - grouped["运费"]
    grouped["结算后利润"] = grouped["结算金额"] - grouped["产品总成本"] - grouped["运费"]

    grouped = grouped.sort_values("GMV", ascending=False)
    grouped = grouped.rename(columns={"货号修正": "货号", cfg.product_name: "商品名称"})
    return grouped


def build_mapping_issues(work: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    main = work[work["是否主商品"]].copy()
    if main.empty:
        return pd.DataFrame()

    issues = main[main["销售规格ID"].isna()].copy()
    if issues.empty:
        return pd.DataFrame(columns=["平台", "店铺名称", "货号", "商品名称", "订单行数", "GMV"])

    result = issues.groupby(
        [cfg.platform, cfg.store_name, "货号修正", cfg.product_name],
        as_index=False
    ).agg(
        订单行数=(cfg.ad_no, "size"),
        GMV=(cfg.paid_amount, "sum"),
    )
    result = result.rename(columns={cfg.platform: "平台", cfg.store_name: "店铺名称", "货号修正": "货号", cfg.product_name: "商品名称"})
    return result.sort_values(["订单行数", "GMV"], ascending=[False, False])


def build_gift_analysis(work: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    gifts = work[work["是否赠品"]].copy()
    if gifts.empty:
        return pd.DataFrame(columns=["赠品名称", "件数", "单件成本", "总成本"])

    rows = []
    for gift_name, unit_cost in GIFT_COST_MAP.items():
        qty = gifts.loc[gifts["赠品名称"] == gift_name, cfg.qty].sum()
        if qty > 0:
            rows.append({
                "赠品名称": gift_name,
                "件数": qty,
                "单件成本": unit_cost,
                "总成本": qty * unit_cost,
            })
    return pd.DataFrame(rows).sort_values("总成本", ascending=False)


def to_excel_bytes(
    summary: Dict[str, float],
    sku_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    gift_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="总览", index=False)
        sku_df.to_excel(writer, sheet_name="货号分析", index=False)
        issue_df.to_excel(writer, sheet_name="映射异常", index=False)
        gift_df.to_excel(writer, sheet_name="赠品分析", index=False)
    return buffer.getvalue()


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


# =========================
# 页面
# =========================
st.title("爱库存经营自动分析工具 v1")
st.caption("上传订单表 + 3 张映射表，自动输出基础经营分析结果。")

with st.sidebar:
    st.header("上传文件")
    order_file = st.file_uploader("1. 订单原始表", type=["xlsx"])
    link_file = st.file_uploader("2. 爱库存艾兰得店铺链接映射表", type=["xlsx"])
    sales_file = st.file_uploader("3. 艾兰得销售规格映射表", type=["xlsx"])
    product_file = st.file_uploader("4. 艾兰得标准产品主档表", type=["xlsx"])
    run_btn = st.button("开始分析", type="primary", use_container_width=True)

st.info("第一版先实现：文件上传、核心指标、货号分析、映射异常、赠品分析、Excel 导出。")

if run_btn:
    missing = []
    if order_file is None:
        missing.append("订单原始表")
    if link_file is None:
        missing.append("店铺链接映射表")
    if sales_file is None:
        missing.append("销售规格映射表")
    if product_file is None:
        missing.append("标准产品主档表")

    if missing:
        st.error("请先上传以下文件：" + "、".join(missing))
        st.stop()

    try:
        order_df = read_excel_any(order_file)
        link_df = read_excel_any(link_file)
        sales_df = read_excel_any(sales_file)
        product_df = read_excel_any(product_file)

        orders, cfg = preprocess_orders(order_df)
        cost_map = build_cost_mapping(link_df, sales_df, product_df)
        enriched = attach_costs(orders, cfg, cost_map)

        summary = calc_metrics(enriched, cfg)
        sku_df = build_sku_analysis(enriched, cfg)
        issue_df = build_mapping_issues(enriched, cfg)
        gift_df = build_gift_analysis(enriched, cfg)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GMV", fmt_money(summary["GMV"]))
        c2.metric("主商品销量", f"{summary['主商品销量']:.0f}")
        c3.metric("结算金额", fmt_money(summary["结算金额"]))
        c4.metric("结算后利润", fmt_money(summary["结算后利润"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("毛利", fmt_money(summary["毛利"]))
        c6.metric("基础佣金", fmt_money(summary["基础佣金"]))
        c7.metric("平台技术服务费", fmt_money(summary["平台技术服务费"]))
        c8.metric("商品佣金率", f"{summary['商品佣金率']:.1%}")

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("高佣加码", fmt_money(summary["高佣加码"]))
        c10.metric("平台交易手续费", fmt_money(summary["平台交易手续费"]))
        c11.metric("运费险", fmt_money(summary["运费险"]))
        c12.metric("流量雷达服务费", fmt_money(summary["流量雷达服务费"]))

        st.subheader("货号分析")
        st.dataframe(
            sku_df.style.format({
                "GMV": "{:,.2f}",
                "销量": "{:,.0f}",
                "营销后回款价": "{:,.2f}",
                "产品总成本": "{:,.2f}",
                "运费": "{:,.2f}",
                "基础佣金": "{:,.2f}",
                "平台技术服务费": "{:,.2f}",
                "高佣加码": "{:,.2f}",
                "商品佣金率": "{:.1%}",
                "运费险": "{:,.2f}",
                "流量雷达服务费": "{:,.2f}",
                "结算金额": "{:,.2f}",
                "毛利": "{:,.2f}",
                "结算后利润": "{:,.2f}",
            }),
            use_container_width=True,
            height=520,
        )

        st.subheader("映射异常")
        if issue_df.empty:
            st.success("未发现主商品映射异常。")
        else:
            st.dataframe(issue_df.style.format({"GMV": "{:,.2f}"}), use_container_width=True)

        st.subheader("赠品分析")
        if gift_df.empty:
            st.info("未识别到赠品。")
        else:
            st.dataframe(
                gift_df.style.format({"单件成本": "{:,.2f}", "总成本": "{:,.2f}"}),
                use_container_width=True,
            )

        excel_bytes = to_excel_bytes(summary, sku_df, issue_df, gift_df)
        st.download_button(
            "下载分析结果 Excel",
            data=excel_bytes,
            file_name="爱库存经营分析结果_v1.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)