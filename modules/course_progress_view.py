# modules/course_progress_view.py

import streamlit as st
from datetime import datetime, date, time, timedelta
from collections import Counter
from streamlit_autorefresh import st_autorefresh

TIME_OPTIONS = ["18:00", "18:30", "20:30", "21:00"]
from .supabase_client import supabase

# テーブルの並び順（course_reservation と合わせる）
TABLE_ORDER = {
    "1-T1": 0,  "1-T2": 1,  "1-T3": 2,  "1-T4": 3,  "1-T5": 4,
    "1-T6": 5,  "1-T7": 6,  "1-T8": 7,  "1-T9": 8,
    "1-C1": 9,  "1-C4": 10, "1-C5": 11, "1-C8": 12, "レコード": 13,
    "2-T1": 14, "2-T2": 15, "2-T3": 16, "2-T4": 17, "2-T5": 18, "2-T6": 19,
    "2-C1": 20, "2-C4": 21, "2-C5": 22, "2-C8": 23,
    "2-R1": 24, "2-R2": 25, "2-R3": 26,
}


def cleanup_old_data():
    """
    今日より前の日付の予約と、それに紐づく course_progress を全削除する。
    （＝前日以前のデータは残さない運用）
    """
    today = date.today()
    start_today = datetime.combine(today, time(0, 0, 0))

    try:
        # 今日より前の予約を取得
        res = (
            supabase.table("course_reservations")
            .select("id, reserved_at")
            .lt("reserved_at", start_today.isoformat())
            .execute()
        )
        old_reservations = res.data or []
        if not old_reservations:
            return  # 消すものなし

        old_ids = [r["id"] for r in old_reservations]

        # 先に進行テーブルを削除（外部キー制約対策）
        supabase.table("course_progress").delete().in_("reservation_id", old_ids).execute()

        # 予約本体を削除
        supabase.table("course_reservations").delete().in_("id", old_ids).execute()

    except Exception as e:
        st.error(f"過去データの削除に失敗しました: {e}")



def parse_dt(dt_str: str):
    """Supabase の TIMESTAMP(+タイムゾーン) を安全に datetime に変換する"""
    if not dt_str:
        return None
    try:
        # 通常の isoformat はまずここで試す
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # "2025-11-18T15:10:20.86786+00:00" などを想定
        # タイムゾーン以降を削り、秒までで切る
        base = dt_str.split("+")[0].split("Z")[0]
        if len(base) > 19:
            base = base[:19]  # "YYYY-MM-DDTHH:MM:SS" まで
        return datetime.fromisoformat(base)



# 予約ステータスを更新（reserved / arrived など）
def set_reservation_status(reservation_id: str, status: str):
    try:
        supabase.table("course_reservations").update(
            {"status": status}
        ).eq("id", reservation_id).execute()
    except Exception as e:
        st.error(f"予約ステータスの更新に失敗しました: {e}")

# 調理フラグを更新（True / False）
def set_cooked_flag(progress_id: str, flag: bool):
    try:
        payload = {"is_cooked": flag}

        if flag:
            # 調理済みにするときは cooked_at も現在時刻でセット
            payload["cooked_at"] = datetime.now().isoformat()
        else:
            # 戻すときは cooked_at をクリア
            payload["cooked_at"] = None

        supabase.table("course_progress").update(payload).eq("id", progress_id).execute()
    except Exception as e:
        st.error(f"調理フラグの更新に失敗しました: {e}")


# 配膳フラグを更新（True / False）
def set_served_flag(progress_id: str, flag: bool):
    try:
        payload = {"is_served": flag}

        if flag:
            # 配膳済みにするときは served_at も現在時刻でセット
            payload["served_at"] = datetime.now().isoformat()
        else:
            # 戻すときは served_at をクリア
            payload["served_at"] = None

        supabase.table("course_progress").update(payload).eq("id", progress_id).execute()
    except Exception as e:
        st.error(f"配膳フラグの更新に失敗しました: {e}")



def fetch_reservations_for_date(target_date: date):
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0, 0))

    res = (
        supabase.table("course_reservations")
        .select("*")
        .gte("reserved_at", start_dt.isoformat())
        .lt("reserved_at", end_dt.isoformat())
        .neq("status", "cancelled")
        .order("reserved_at", desc=False)
        .execute()
    )
    return res.data or []


def fetch_progress_for_reservations(reservation_ids):
    if not reservation_ids:
        return []

    res = (
        supabase.table("course_progress")
        .select("*")
        .in_("reservation_id", reservation_ids)
        .order("scheduled_time", desc=False)
        .execute()
    )
    return res.data or []



def fetch_items_for_ids(item_ids):
    if not item_ids:
        return {}

    res = (
        supabase.table("course_items")
        .select("id, item_name, offset_minutes")
        .in_("id", item_ids)
        .execute()
    )
    items = res.data or []
    return {i["id"]: i for i in items}


def update_reservation_arrived(reservation_id):
    now_iso = datetime.now().isoformat()
    supabase.table("course_reservations").update(
        {"status": "arrived", "arrived_at": now_iso}
    ).eq("id", reservation_id).execute()


def update_cooked(progress_id):
    now_iso = datetime.now().isoformat()
    supabase.table("course_progress").update(
        {"is_cooked": True, "cooked_at": now_iso}
    ).eq("id", progress_id).execute()


def update_served(progress_id):
    # 既存コードとの互換用ヘルパー（配膳済みにする）
    set_served_flag(progress_id, True)


def show_board():
    cleanup_old_data()

    # ---- 自動更新 ON/OFF ----
    if "auto_refresh_board" not in st.session_state:
        st.session_state["auto_refresh_board"] = True  # デフォルトON

    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.session_state["auto_refresh_board"] = st.checkbox(
            "自動更新",
            value=st.session_state["auto_refresh_board"],
            help="1〜2秒ごとに最新の状態を反映します",
        )

    if st.session_state["auto_refresh_board"]:
        # 2000ms = 2秒ごとにスクリプトを再実行（画面はほぼそのまま）
        st_autorefresh(interval=5000, key="board_autorefresh_counter")

    col_date, col_info = st.columns([1, 1])
    with col_date:
        target_date = st.date_input("対象日", value=date.today())
    with col_info:
        st.caption("※ 予約数が多い日は、画面下の横スクロールバーで左右に移動できます。")

    reservations = fetch_reservations_for_date(target_date)
    if not reservations:
        st.info("該当日のコース予約はありません。")
        return
    
    # 時間 → テーブル順で並べ替え
    def sort_key_resv(r):
        dt = datetime.fromisoformat(r["reserved_at"])
        table = r.get("table_no") or ""
        table_idx = TABLE_ORDER.get(table, 999)
        return (dt, table_idx)

    reservations = sorted(reservations, key=sort_key_resv)

    # ここでコンテナの横幅を「予約数 × 300px」で決める（初期値）
    per_card_width = 300
    width_px = max(300, per_card_width * len(reservations))

    st.markdown(f"""
    <style>
    .block-container {{
        min-width: {width_px}px;
        margin-left: 0 !important;
        margin-right: auto !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # # ======================
    # # 予約件数 + 時間帯別 + コース別の集計表示
    # # ======================

    # # 総件数
    # total_count = len(reservations)

    # # 時間帯ごとの件数
    # time_counter = Counter()
    # for r in reservations:
    #     dt = datetime.fromisoformat(r["reserved_at"])
    #     time_str = dt.strftime("%H:%M")
    #     if time_str in TIME_OPTIONS:
    #         time_counter[time_str] += 1

    # # コース情報取得（id → name）
    # res_courses = (
    #     supabase.table("course_master")
    #     .select("id, name")
    #     .execute()
    # )
    # course_rows = res_courses.data or []
    # course_map = {c["id"]: c["name"] for c in course_rows}

    # # コース別件数
    # course_counter = Counter()
    # for r in reservations:
    #     cid = r.get("course_id")
    #     if cid:
    #         course_counter[cid] += 1

    # # 表示（予約登録画面と同じフォーマット）
    # st.markdown(
    #     f"#### {target_date.strftime('%Y/%m/%d')} の予約件数：**{total_count}件**"
    # )

    # # 時間帯別
    # time_lines = []
    # for slot in TIME_OPTIONS:  # ["18:00", "18:30", "20:30", "21:00"]
    #     count = time_counter.get(slot, 0)
    #     time_lines.append(f"{slot}: {count}件")
    # st.caption("時間帯別：" + " / ".join(time_lines))

    # # コース別
    # course_lines = []
    # for cid, cnt in course_counter.items():
    #     name = course_map.get(cid, "不明なコース")
    #     course_lines.append(f"{name}: {cnt}件")
    # if course_lines:
    #     st.caption("コース別：" + " / ".join(course_lines))

    # ===== ここからボード表示のための progress 集計 =====
    reservation_ids = [r["id"] for r in reservations]
    progress_rows = fetch_progress_for_reservations(reservation_ids)

    # item_id → item 情報
    item_ids = list({p["course_item_id"] for p in progress_rows})
    item_map = fetch_items_for_ids(item_ids)

    # 予約ごとの progress 集計
    #   has_any_progress: 何か1つでも商品が存在する予約
    #   has_unserved:     未配膳の商品が1つでも残っている予約
    #   progress_by_res:  未配膳の商品だけを予約ごとにまとめたもの（表示用）
    progress_by_res = {}
    has_any_progress = set()
    has_unserved = set()

    for p in progress_rows:
        rid = p["reservation_id"]
        has_any_progress.add(rid)

        if not p.get("is_served", False):
            has_unserved.add(rid)
            progress_by_res.setdefault(rid, []).append(p)

    # ★ アクティブな予約：
    #   1) まだ progress 自体が1つもない予約（これから商品が出てくる予約）
    #   2) progress はあるが、未配膳の商品が1つでも残っている予約
    #   → つまり「全商品が配膳済みの予約」だけを除外する
    active_reservations = [
        r for r in reservations
        if (r["id"] not in has_any_progress) or (r["id"] in has_unserved)
    ]

    if not active_reservations:
        # 全予約が「全商品配膳済み」の場合
        st.info("配膳待ちのコース商品はありません。")
        return

    # アクティブ予約数に合わせて横幅を再計算（上書き）
    width_px = max(300, per_card_width * len(active_reservations))
    st.markdown(f"""
    <style>
    .block-container {{
        min-width: {width_px}px;
        margin-left: 0 !important;
        margin-right: auto !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 予約順に並べてカラム表示（アクティブな予約のみ）
    cols = st.columns(len(active_reservations))

    for idx, resv in enumerate(active_reservations):
        with cols[idx]:
            resv_id = resv["id"]
            resv_time = datetime.fromisoformat(resv["reserved_at"])
            items_for_res = sorted(
                progress_by_res.get(resv_id, []),
                key=lambda x: x["scheduled_time"],
            )

            # ===== 見出し：時間 / 名前＋人数 / テーブル =====
            guest_name = (resv.get("guest_name") or "お名前未入力")
            guest_count = resv.get("guest_count") or "-"
            table_label = f"{resv.get('table_no') or '-'}"
            st.markdown(
                f"""
                <div style="
                    background-color:#f2f2f2;
                    border-radius:10px;
                    padding:10px 4px;
                    text-align:center;
                    font-weight:600;
                    font-size:18px;
                    margin-bottom:8px;
                ">
                    <div style="color:#d9534f; font-weight:700; font-size:20px;">
                        {resv_time.strftime('%H:%M')}
                    </div>
                    <div>
                        {guest_name} 様（{guest_count} 名）
                    </div>
                    <div style="
                        font-weight:700;
                        font-size:20px;
                        margin-bottom:8px;
                        text-align: center;
                        color: #d9534f;
                    ">
                        {table_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 来店済みボタン（トグル式）
            current_status = resv.get("status") or "reserved"

            if current_status != "arrived":
                if st.button("来店済みにする", key=f"arrived_{idx}_{resv_id}"):
                    set_reservation_status(resv_id, "arrived")
                    st.rerun()
            else:
                st.success("来店済み")
                if st.button("来店済みを取り消す", key=f"undo_arrived_{idx}_{resv_id}"):
                    set_reservation_status(resv_id, "reserved")
                    st.rerun()

            st.markdown("---")

            # items_for_res が空＝
            #   ・まだ progress が1つもない予約（これから商品が出てくる）
            #   → カードは残しつつメッセージだけ表示
            if not items_for_res:
                st.caption("※ この予約には、表示可能な商品がありません。")
                continue

            # ===== 各商品の行（未配膳のものだけ） =====
            total_items = len(items_for_res)

            for row_idx, p in enumerate(items_for_res):
                item = item_map.get(p["course_item_id"])
                if not item:
                    continue

                sched_time = datetime.fromisoformat(p["scheduled_time"])
                time_str = sched_time.strftime('%H:%M')

                is_cooked = p.get("is_cooked", False)
                is_served = p.get("is_served", False)  # ここは全部 False のはずだが念のため

                # メイン枠なら、予約ごとのメイン料理名で上書き
                display_name = item["item_name"]
                if item["item_name"] == "メイン":
                    # ★ 種類ごとに別レコードになった main_detail / quantity を優先
                    detail = p.get("main_detail")
                    qty = p.get("quantity", 1)
                    if detail:
                        display_name = f"{detail}：{qty}"
                    else:
                        # 旧データ用フォールバック（main_choice 全体の文字列）
                        main_choice = resv.get("main_choice")
                        if main_choice:
                            display_name = main_choice

                # 商品見出し：時間(赤)＋商品名
                st.markdown(
                    f"""
                    <div style="margin-top:4px; margin-bottom:4px;">
                        <div style="font-size:16px; font-weight:600;">
                            <span style="color:#d9534f; font-weight:700; margin-right:4px;">
                                {time_str}
                            </span>
                            <span>{display_name}</span>
                        </div>
                        <div style="font-size:16px; color:#6495ED; margin-left:2px; font-weight:bold;">
                            テーブル：{resv.get('table_no') or '-'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ボタン行
                c1, c2 = st.columns(2)
                with c1:
                    if not is_cooked:
                        if st.button("調理済み", key=f"cook_{idx}_{row_idx}_{p['id']}"):
                            set_cooked_flag(p["id"], True)
                            st.rerun()
                    else:
                        st.error("調理済み")
                        if st.button("調理済みを戻す", key=f"undo_cook_{idx}_{row_idx}_{p['id']}"):
                            set_cooked_flag(p["id"], False)
                            st.rerun()

                with c2:
                    if not is_served:
                        if st.button("配膳済み", key=f"serve_{idx}_{row_idx}_{p['id']}"):
                            update_served(p["id"])
                            st.rerun()

                # 商品と商品の間の区切り線
                if row_idx < total_items - 1:
                    st.markdown(
                        "<hr style='margin:8px 0; border:none; border-top:1px solid #333333;'/>",
                        unsafe_allow_html=True
                    )







# ===== 調理済み・配膳済み一覧 =====

def show_cooked_list():
    cleanup_old_data()
    st.subheader("調理済み一覧")

    if "auto_refresh_cooked" not in st.session_state:
        st.session_state["auto_refresh_cooked"] = True

    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.session_state["auto_refresh_cooked"] = st.checkbox(
            "自動更新",
            value=st.session_state["auto_refresh_cooked"],
            key="chk_auto_refresh_cooked",
        )

    if st.session_state["auto_refresh_cooked"]:
        st_autorefresh(interval=5000, key="cooked_autorefresh_counter")

    target_date = st.date_input("対象日（調理日）", value=date.today(), key="cooked_date")
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0, 0))

    # この日に「調理済み」になったものだけ取得
    res = (
        supabase.table("course_progress")
        .select("*")
        .eq("is_cooked", True)
        .gte("cooked_at", start_dt.isoformat())
        .lt("cooked_at", end_dt.isoformat())
        .order("scheduled_time", desc=False)
        .execute()
    )
    rows = res.data or []
    if not rows:
        st.info("該当日の調理済みデータはありません。")
        return

    # 関連する予約・商品情報を取得
    reservation_ids = list({r["reservation_id"] for r in rows})
    item_ids = list({r["course_item_id"] for r in rows})

    # 予約情報
    res_resv = (
        supabase.table("course_reservations")
        .select("id, reserved_at, guest_name, guest_count, table_no, main_choice")
        .in_("id", reservation_ids)
        .execute()
    )
    reservations = res_resv.data or []
    if not reservations:
        st.info("該当する予約データがありません。")
        return

    # 商品マスタ
    item_map = fetch_items_for_ids(item_ids)

    # 予約ごとに progress をまとめる
    progress_by_res = {}
    for r in rows:
        progress_by_res.setdefault(r["reservation_id"], []).append(r)

    # 「調理済みが1つでもある予約」のみに絞る
    reservations = [r for r in reservations if r["id"] in progress_by_res]

    # 予約を「予約時間 → テーブル順」で並べ替え（進行ボードと同じ）
    def sort_key_resv(r):
        dt = datetime.fromisoformat(r["reserved_at"])
        table = r.get("table_no") or ""
        table_idx = TABLE_ORDER.get(table, 999)
        return (dt, table_idx)

    reservations = sorted(reservations, key=sort_key_resv)

    # 横スクロールできるように幅を調整（進行ボードと同じ考え方）
    per_card_width = 300
    width_px = max(300, per_card_width * len(reservations))
    st.markdown(
        f"""
        <style>
        .block-container {{
            min-width: {width_px}px;
            margin-left: 0 !important;
            margin-right: auto !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 予約単位でカラムを作成（左から順番に並ぶ）
    cols = st.columns(len(reservations))

    for idx, resv in enumerate(reservations):
        with cols[idx]:
            resv_time = datetime.fromisoformat(resv["reserved_at"])
            guest_name = (resv.get("guest_name") or "お名前未入力")
            guest_count = resv.get("guest_count") or "-"
            table_no = resv.get("table_no") or "-"

            # 予約ヘッダー（時間＋名前＋人数）※来店済み表記は無し
            st.markdown(
                f"""
                <div style="
                    background-color:#f2f2f2;
                    border-radius:10px;
                    padding:10px 4px;
                    text-align:center;
                    font-weight:600;
                    font-size:18px;
                    margin-bottom:8px;
                ">
                    <div style="color:#d9534f; font-weight:700; font-size:20px;">
                        {resv_time.strftime('%H:%M')}
                    </div>
                    <div>
                        {guest_name} 様（{guest_count} 名）
                    </div>
                    <div style="font-size:20px; color:#d9534f; margin-left:2px; font-weight:bold;">
                        {resv.get('table_no') or '-'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            items_for_res = sorted(
                progress_by_res.get(resv["id"], []),
                key=lambda x: x["scheduled_time"],
            )
            if not items_for_res:
                st.caption("※ この予約の調理済み商品はありません。")
                continue

            total_items = len(items_for_res)

            # この予約で「調理済み」になっている商品だけを表示
            for row_idx, p in enumerate(items_for_res):
                item = item_map.get(p["course_item_id"])
                if not item:
                    continue

                cooked_at = parse_dt(p["cooked_at"])

                # メイン枠なら、予約ごとのメイン料理名で上書き
                display_name = item["item_name"]
                if item["item_name"] == "メイン":
                    detail = p.get("main_detail")
                    qty = p.get("quantity", 1)
                    if detail:
                        display_name = f"{detail}：{qty}"
                    else:
                        main_choice = resv.get("main_choice")
                        if main_choice:
                            display_name = main_choice


                st.markdown(
                    f"""
                    <div style="margin-top:4px; margin-bottom:4px;">
                        <div style="font-size:16px; font-weight:600;">
                            <span style="color:#d9534f; font-weight:700; margin-right:4px;">
                                {cooked_at.strftime('%H:%M')}
                            </span>
                            <span>{display_name}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 区切り線
                if row_idx < total_items - 1:
                    st.markdown(
                        "<hr style='margin:8px 0; border:none; border-top:1px solid #333333;'/>",
                        unsafe_allow_html=True
                    )



def show_served_list():
    cleanup_old_data()
    st.subheader("配膳済み一覧")

    if "auto_refresh_cooked" not in st.session_state:
        st.session_state["auto_refresh_cooked"] = True

    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.session_state["auto_refresh_cooked"] = st.checkbox(
            "自動更新",
            value=st.session_state["auto_refresh_cooked"],
            key="chk_auto_refresh_cooked",
        )

    if st.session_state["auto_refresh_cooked"]:
        st_autorefresh(interval=5000, key="cooked_autorefresh_counter")

    target_date = st.date_input("対象日（配膳日）", value=date.today(), key="served_date")
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0, 0))

    # この日に「配膳済み」になったものだけ取得
    res = (
        supabase.table("course_progress")
        .select("*")
        .eq("is_served", True)
        .gte("served_at", start_dt.isoformat())
        .lt("served_at", end_dt.isoformat())
        .order("scheduled_time", desc=False)
        .execute()
    )
    rows = res.data or []
    if not rows:
        st.info("該当日の配膳済みデータはありません。")
        return

    # 関連する予約・商品情報を取得
    reservation_ids = list({r["reservation_id"] for r in rows})
    item_ids = list({r["course_item_id"] for r in rows})

    # 予約情報
    res_resv = (
        supabase.table("course_reservations")
        .select("id, reserved_at, guest_name, guest_count, table_no, main_choice")
        .in_("id", reservation_ids)
        .execute()
    )
    reservations = res_resv.data or []
    if not reservations:
        st.info("該当する予約データがありません。")
        return

    # 商品マスタ
    item_map = fetch_items_for_ids(item_ids)

    # 予約ごとに progress をまとめる
    progress_by_res = {}
    for r in rows:
        progress_by_res.setdefault(r["reservation_id"], []).append(r)

    # 「配膳済みが1つでもある予約」のみに絞る
    reservations = [r for r in reservations if r["id"] in progress_by_res]

    # 予約を「予約時間 → テーブル順」で並べ替え（進行ボードと同じ）
    def sort_key_resv(r):
        dt = datetime.fromisoformat(r["reserved_at"])
        table = r.get("table_no") or ""
        table_idx = TABLE_ORDER.get(table, 999)
        return (dt, table_idx)

    reservations = sorted(reservations, key=sort_key_resv)

    # 横スクロールできるように幅を調整（進行ボードと同じ考え方）
    per_card_width = 300
    width_px = max(300, per_card_width * len(reservations))
    st.markdown(
        f"""
        <style>
        .block-container {{
            min-width: {width_px}px;
            margin-left: 0 !important;
            margin-right: auto !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 予約単位でカラムを作成（左から順番に並ぶ）
    cols = st.columns(len(reservations))

    for idx, resv in enumerate(reservations):
        with cols[idx]:
            resv_time = datetime.fromisoformat(resv["reserved_at"])
            guest_name = (resv.get("guest_name") or "お名前未入力")
            guest_count = resv.get("guest_count") or "-"
            table_no = resv.get("table_no") or "-"

            # 予約ヘッダー（時間＋名前＋人数＋テーブル）※ステータス表記なし
            st.markdown(
                f"""
                <div style="
                    background-color:#f2f2f2;
                    border-radius:10px;
                    padding:10px 4px;
                    text-align:center;
                    font-weight:600;
                    font-size:18px;
                    margin-bottom:8px;
                ">
                    <div style="color:#d9534f; font-weight:700; font-size:20px;">
                        {resv_time.strftime('%H:%M')}
                    </div>
                    <div>
                        {guest_name} 様（{guest_count} 名）
                    </div>
                    <div style="font-size:20px; color:#d9534f; margin-left:2px; font-weight:bold;">
                        {table_no}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            items_for_res = sorted(
                progress_by_res.get(resv["id"], []),
                key=lambda x: x["scheduled_time"],
            )
            if not items_for_res:
                st.caption("※ この予約の配膳済み商品はありません。")
                continue

            total_items = len(items_for_res)

            # この予約で「配膳済み」になっている商品だけを表示
            for row_idx, p in enumerate(items_for_res):
                item = item_map.get(p["course_item_id"])
                if not item:
                    continue

                served_at = parse_dt(p["served_at"])

                # メイン枠なら、予約ごとのメイン料理名で上書き
                display_name = item["item_name"]
                if item["item_name"] == "メイン":
                    detail = p.get("main_detail")
                    qty = p.get("quantity", 1)
                    if detail:
                        display_name = f"{detail}：{qty}"
                    else:
                        main_choice = resv.get("main_choice")
                        if main_choice:
                            display_name = main_choice


                st.markdown(
                    f"""
                    <div style="margin-top:4px; margin-bottom:4px;">
                        <div style="font-size:16px; font-weight:600;">
                            <span style="color:#d9534f; font-weight:700; margin-right:4px;">
                                {served_at.strftime('%H:%M')}
                            </span>
                            <span>{display_name}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ★ここで「配膳済みを戻す」ボタン（→ 進行ボードへ戻す）
                if st.button(
                    "配膳済みを戻す",
                    key=f"undo_served_{idx}_{row_idx}_{p['id']}",
                ):
                    set_served_flag(p["id"], False)
                    st.rerun()

                # 区切り線
                if row_idx < total_items - 1:
                    st.markdown(
                        "<hr style='margin:8px 0; border:none; border-top:1px solid #333333;'/>",
                        unsafe_allow_html=True
                    )

