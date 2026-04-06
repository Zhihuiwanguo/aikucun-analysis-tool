import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="爱库存经营自动分析工具 v1", layout="wide")


# =====================================
# 固定业务口径
# =====================================
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


# =====================================
# 通用工具
# =====================================
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


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


# =====================================
# 字段配置
# =====================================
@dataclass
class ColumnConfig:
    platform: Optional[str]
    store_name: str
    ad_no: str
    order_status: str
    aftersale_type: Optional[str]
    pay_time: Optional[str]
    sku_code: Optional[str]
    sales_spec_id: Optional[str]
    barcode: Optional[str]
    product_name: str
    spec_text: Optional[str]
    qty: str
    paid_amount: str
    marketing_return: str
    activity_name: Optional[str]


# =====================================
# 字段识别
# =====================================
def detect_order_columns(df: pd.DataFrame) -> ColumnConfig:
    return ColumnConfig(
        platform=pick_col(df, ["平台"], required=False),
        store_name=pick_col(df, ["店铺名称"]),
        ad_no=pick_col(df, ["ad单号", "AD单号", "ad单", "母单号"]),
        order_status=pick_col(df, ["订单状态"]),
        aftersale_type=pick_col(df, ["售后类型"], required=False),
        pay_time=pick_col(df, ["支付时间", "付款时间", "支付成功时间"], required=False),
        sku_code=pick_col(df, ["货号"], required=False),
        sales_spec_id=pick_col(df, ["销售规格ID"], required=False),
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


# =====================================
# 业务预处理
# =====================================
def mark_gift_name(name: str) -> Tuple[bool, str]:
    text = str(name).strip()
    if "维C赠品" in text:
        return True, "维C赠品"
    if "运动水杯" in text or "水杯" in text:
        return True, "运动水杯"
    if "行李箱" in text:
        return True, "小熊行李箱"
    return False, ""


def preprocess_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnConfig]:
    cfg = detect_order_columns(df)
    work = df.copy()

    # 平台
    if cfg.platform:
        work[cfg.platform] = as_str(work[cfg.platform])
    else:
        work["平台"] = "爱库存"
        cfg.platform = "平台"

    # 基础文本列
    for col in [cfg.store_name, cfg.ad_no, cfg.order_status, cfg.product_name]:
        work[col] = as_str(work[col])

    # 可选文本列
    if cfg.aftersale_type:
        work[cfg.aftersale_type] = as_str(work[cfg.aftersale_type])

    if cfg.sku_code:
        work[cfg.sku_code] = as_str(work[cfg.sku_code])
    else:
        work["货号"] = ""
        cfg.sku_code = "货号"

    if cfg.sales_spec_id:
        work[cfg.sales_spec_id] = as_str(work[cfg.sales_spec_id])
    else:
        work["销售规格ID"] = ""
        cfg.sales_spec_id = "销售规格ID"

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

    # 数值列
    work[cfg.qty] = as_num(work[cfg.qty])
    work[cfg.paid_amount] = as_num(work[cfg.paid_amount])
    work[cfg.marketing_return] = as_num(work[cfg.marketing_return])

    # 赠品识别
    gift_flags = work[cfg.product_name].apply(mark_gift_name)
    work["赠品名称"] = gift_flags.apply(lambda x: x[1])
    work["是否赠品"] = gift_flags.apply(lambda x: x[0]) | as_str(work[cfg.sku_code]).eq("")

    # 空货号赠品：货号修正为条形码
    work["货号修正"] = as_str(work[cfg.sku_code])
    empty_sku_mask = work["货号修正"].eq("") & work["是否赠品"]
    work.loc[empty_sku_mask, "货号修正"] = as_str(work.loc[empty_sku_mask, cfg.barcode])

    # 订单表内销售规格ID
    work["订单销售规格ID"] = as_str(work[cfg.sales_spec_id])

    # 有效订单
    if cfg.aftersale_type:
        aftersale_series = as_str(work[cfg.aftersale_type])
    else:
        aftersale_series = pd.Series("", index=work.index)

    work["是否无效订单"] = (
        as_str(work[cfg.order_status]).eq("订单取消")
        | aftersale_series.eq("退货退款")
    )
    work["是否有效订单"] = ~work["是否无效订单"]

    # 主商品
    work["是否主商品"] = work["是否有效订单"] & (~work["是否赠品"])

    return work, cfg


# =====================================
# 映射构建
# =====================================
def build_link_map(link_df: pd.DataFrame) -> pd.DataFrame:
    link = link_df.copy()
    link = clean_columns(link)

    for col in ["平台", "店铺名称", "货号", "销售规格ID"]:
        link[col] = as_str(link[col])

    keep_cols = ["平台", "店铺名称", "货号", "销售规格ID"]
    for optional_col in ["条形码", "规格", "销售规格名称", "是否主成交规格"]:
        if optional_col in link.columns:
            keep_cols.append(optional_col)

    link = link[keep_cols].drop_duplicates()
    return link


def build_sales_cost_map(sales_df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    sales = sales_df.copy()
    product = product_df.copy()

    sales = clean_columns(sales)
    product = clean_columns(product)

    for col in ["销售规格ID", "标准产品ID"]:
        sales[col] = as_str(sales[col])

    for col in ["标准产品ID", "标准产品名称"]:
        product[col] = as_str(product[col])

    sales["产品总成本"] = as_num(sales["产品总成本"])

    sales_keep = ["销售规格ID", "标准产品ID", "产品总成本"]
    for optional_col in ["销售规格名称", "销售数量"]:
        if optional_col in sales.columns:
            sales_keep.append(optional_col)

    sales = sales[sales_keep].drop_duplicates()
    product = product[["标准产品ID", "标准产品名称"]].drop_duplicates()

    merged = sales.merge(product, on="标准产品ID", how="left")
    merged = merged.rename(columns={"产品总成本": "映射产品总成本"})
    return merged


def attach_costs(
    orders: pd.DataFrame,
    cfg: ColumnConfig,
    link_map: pd.DataFrame,
    sales_cost_map: pd.DataFrame,
) -> pd.DataFrame:
    work = orders.copy()

    # 1. 先用 平台+店铺名称+货号 从链接表补销售规格ID
    work = work.merge(
        link_map[["平台", "店铺名称", "货号", "销售规格ID"]],
        left_on=[cfg.platform, cfg.store_name, "货号修正"],
        right_on=["平台", "店铺名称", "货号"],
        how="left",
        suffixes=("", "_link"),
    )

    # 2. 最终销售规格ID：订单表优先，链接表兜底
    work["最终销售规格ID"] = as_str(work["订单销售规格ID"])
    need_fill_spec = work["最终销售规格ID"].eq("")
    work.loc[need_fill_spec, "最终销售规格ID"] = as_str(work.loc[need_fill_spec, "销售规格ID"])

    # 3. 按最终销售规格ID去销售规格映射表带成本
    work = work.merge(
        sales_cost_map,
        left_on="最终销售规格ID",
        right_on="销售规格ID",
        how="left",
        suffixes=("", "_cost"),
    )

    # 4. 统一关键列
    work["销售规格ID"] = as_str(work["最终销售规格ID"])
    work["产品总成本"] = as_num(work["映射产品总成本"])
    work["运费"] = 0.0

    # 5. 运费按 ad 单号分摊到主商品
    main_ad_counts = (
        work[work["是否主商品"]]
        .groupby(cfg.ad_no, dropna=False)
        .size()
        .rename("主商品行数")
        .reset_index()
    )
    work = work.merge(main_ad_counts, on=cfg.ad_no, how="left")
    work["主商品行数"] = work["主商品行数"].fillna(0)

    alloc_mask = work["是否主商品"] & (work["主商品行数"] > 0)
    work.loc[alloc_mask, "运费"] = FREIGHT_PER_AD / work.loc[alloc_mask, "主商品行数"]

    return work


# =====================================
# 指标计算
# =====================================
def calc_metrics(work: pd.DataFrame, cfg: ColumnConfig) -> Dict[str, float]:
    main = work[work["是否主商品"]].copy()
    gifts = work[work["是否赠品"]].copy()

    gmv = main[cfg.paid_amount].sum()
    qty = main[cfg.qty].sum()
    ad_cnt = float(main[cfg.ad_no].nunique())

    marketing_return = main[cfg.marketing_return].sum()
    cost_total = main["产品总成本"].sum()
    freight_total = main["运费"].sum()

    gross_profit = gmv - cost_total - freight_total

    base_commission = gmv * BASE_COMMISSION_RATE
    tech_service_fee = gmv * TECH_SERVICE_RATE
    high_commission = gmv - marketing_return - tech_service_fee - base_commission
    commission_rate = safe_div(base_commission + high_commission, gmv)

    transaction_fee = gmv * TRANSACTION_FEE_RATE
    shipping_insurance = ad_cnt * SHIPPING_INSURANCE_PER_AD
    flow_radar_fee = gmv * FLOW_RADAR_RATE

    settlement_amount = marketing_return - shipping_insurance - flow_radar_fee
    settlement_profit = settlement_amount - cost_total - freight_total

    gift_cost = 0.0
    for gift_name, unit_cost in GIFT_COST_MAP.items():
        gift_qty = gifts.loc[gifts["赠品名称"] == gift_name, cfg.qty].sum()
        gift_cost += gift_qty * unit_cost

    return {
        "GMV": gmv,
        "主商品销量": qty,
        "有效主商品订单数": ad_cnt,
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
    grouped["商品佣金率"] = (
        (grouped["基础佣金"] + grouped["高佣加码"])
        / grouped["GMV"].replace(0, pd.NA)
    )

    total_gmv = grouped["GMV"].sum()
    total_shipping_insurance = main[cfg.ad_no].nunique() * SHIPPING_INSURANCE_PER_AD
    if total_gmv > 0:
        grouped["运费险"] = grouped["GMV"] / total_gmv * total_shipping_insurance
    else:
        grouped["运费险"] = 0.0

    grouped["流量雷达服务费"] = grouped["GMV"] * FLOW_RADAR_RATE
    grouped["结算金额"] = grouped["营销后回款价"] - grouped["运费险"] - grouped["流量雷达服务费"]
    grouped["毛利"] = grouped["GMV"] - grouped["产品总成本"] - grouped["运费"]
    grouped["结算后利润"] = grouped["结算金额"] - grouped["产品总成本"] - grouped["运费"]

    grouped = grouped.rename(
        columns={
            "货号修正": "货号",
            cfg.product_name: "商品名称",
        }
    )
    grouped = grouped.sort_values("GMV", ascending=False)
    return grouped


def build_mapping_issues(work: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    main = work[work["是否主商品"]].copy()
    if main.empty:
        return pd.DataFrame(columns=["平台", "店铺名称", "货号", "商品名称", "问题类型", "订单行数", "GMV"])

    issue_frames = []

    # 问题1：缺少最终销售规格ID
    no_spec = main[main["销售规格ID"].fillna("").eq("")]
    if not no_spec.empty:
        tmp = no_spec.groupby(
            [cfg.platform, cfg.store_name, "货号修正", cfg.product_name],
            as_index=False
        ).agg(
            订单行数=(cfg.ad_no, "size"),
            GMV=(cfg.paid_amount, "sum"),
        )
        tmp["问题类型"] = "缺少销售规格ID"
        issue_frames.append(tmp)

    # 问题2：有销售规格ID，但没映射到标准产品
    no_product = main[
        (~main["销售规格ID"].fillna("").eq(""))
        & (main["标准产品ID"].fillna("").eq(""))
    ]
    if not no_product.empty:
        tmp = no_product.groupby(
            [cfg.platform, cfg.store_name, "货号修正", cfg.product_name],
            as_index=False
        ).agg(
            订单行数=(cfg.ad_no, "size"),
            GMV=(cfg.paid_amount, "sum"),
        )
        tmp["问题类型"] = "销售规格ID无法映射标准产品"
        issue_frames.append(tmp)

    if not issue_frames:
        return pd.DataFrame(columns=["平台", "店铺名称", "货号", "商品名称", "问题类型", "订单行数", "GMV"])

    result = pd.concat(issue_frames, ignore_index=True)
    result = result.rename(
        columns={
            cfg.platform: "平台",
            cfg.store_name: "店铺名称",
            "货号修正": "货号",
            cfg.product_name: "商品名称",
        }
    )
    result = result.sort_values(["问题类型", "订单行数", "GMV"], ascending=[True, False, False])
    return result


def build_gift_analysis(work: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    gifts = work[work["是否赠品"]].copy()
    if gifts.empty:
        return pd.DataFrame(columns=["赠品名称", "件数", "单件成本", "总成本"])

    rows = []
    for gift_name, unit_cost in GIFT_COST_MAP.items():
        qty = gifts.loc[gifts["赠品名称"] == gift_name, cfg.qty].sum()
        if qty > 0:
            rows.append(
                {
                    "赠品名称": gift_name,
                    "件数": qty,
                    "单件成本": unit_cost,
                    "总成本": qty * unit_cost,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["赠品名称", "件数", "单件成本", "总成本"])

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


# =====================================
# 页面
# =====================================
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

        validate_mapping_columns(link_df, sales_df, product_df)

        orders, cfg = preprocess_orders(order_df)
        link_map = build_link_map(link_df)
        sales_cost_map = build_sales_cost_map(sales_df, product_df)
        enriched = attach_costs(orders, cfg, link_map, sales_cost_map)

        summary = calc_metrics(enriched, cfg)
        sku_df = build_sku_analysis(enriched, cfg)
        issue_df = build_mapping_issues(enriched, cfg)
        gift_df = build_gift_analysis(enriched, cfg)

        # 核心指标
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

        c13, c14 = st.columns(2)
        c13.metric("有效主商品订单数", f"{summary['有效主商品订单数']:.0f}")
        c14.metric("赠品总成本", fmt_money(summary["赠品总成本"]))

        st.subheader("货号分析")
        if sku_df.empty:
            st.info("暂无主商品数据。")
        else:
            st.dataframe(
                sku_df.style.format(
                    {
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
                    }
                ),
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