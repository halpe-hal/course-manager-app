# modules/course_reservation.py

import streamlit as st
from datetime import datetime, date, time, timedelta
from collections import Counter
from .supabase_client import supabase
from typing import Optional, Tuple
from .time_utils import get_today_jst

import pandas as pd
import re

# テーブル番号の選択肢
TABLE_OPTIONS = [
    "1-T1", "1-T2", "1-T3", "1-T4", "1-T5", "1-T6", "1-T7", "1-T8", "1-T9",
    "1-C1", "1-C4", "1-C5", "1-C8", "レコード",
    "2-T1", "2-T2", "2-T3", "2-T4", "2-T5", "2-T6",
    "2-C1", "2-C4", "2-C5", "2-C8",
    "2-R1", "2-R2", "2-R3",
]

# テーブルの並び順マップ（0,1,2,... のインデックス）
TABLE_ORDER = {t: i for i, t in enumerate(TABLE_OPTIONS)}

# 予約時間の選択肢（固定）
TIME_OPTIONS = ["18:00", "18:30", "20:30", "21:00"]

MAIN_OPTIONS = [
    "パスタ",
    "ピザ",
]


# =========================
# メイン内訳系ヘルパー
# =========================
def parse_main_choice_to_counts(main_choice: Optional[str]):
    """
    'パスタ：1、ピザ：2' のような文字列を
    {'パスタ': 1, 'ピザ': 2} に変換する。
    """
    counts = {name: 0 for name in MAIN_OPTIONS}
    if not main_choice:
        return counts

    s = str(main_choice)
    for part in s.split("、"):
        part = part.strip()
        if not part:
            continue

        if "：" in part:
            label, num = part.split("：", 1)
        elif ":" in part:
            label, num = part.split(":", 1)
        else:
            continue

        label = label.strip()
        try:
            n = int(num.strip())
        except ValueError:
            continue

        if label in counts:
            counts[label] = n

    return counts


def counts_to_main_choice(counts) -> Optional[str]:
    """
    {'パスタ': 1, 'ピザ': 2} を 'パスタ：1、ピザ：2' に戻す。
    どれも0なら None を返す。
    """
    parts = [f"{name}：{cnt}" for name, cnt in counts.items() if cnt > 0]
    return "、".join(parts) if parts else None


def _default_main_distribution(guest_count: int):
    """
    パスタ/ピザのデフォルト配分。
    2名予約は1つずつ、3名予約はパスタ1・ピザ2、4名予約はパスタ2・ピザ2、
    5名予約はパスタ3・ピザ2、それ以外はざっくり半分。
    """
    if guest_count <= 0:
        return 0, 0
    if guest_count == 1:
        return 1, 0
    if guest_count == 2:
        return 1, 1
    if guest_count == 3:
        return 1, 2
    if guest_count == 4:
        return 2, 2
    if guest_count == 5:
        return 3, 2

    # 6名以上はざっくり半分
    pasta = guest_count // 2
    pizza = guest_count - pasta
    return pasta, pizza


def parse_main_counts_from_text(guest_count: int, text: str):
    """
    ブロック内のテキストからパスタ/ピザの人数を推定する。

    - 「プラン名：〜」の行はカウント対象外
    - 「プラン名」行の次の1行もカウント対象外（説明が続いているため）
    - 回答部分に「ピザ」のみ → 全員ピザ
    - 回答部分に「パスタ」のみ → 全員パスタ
    - 両方出現 → 出現回数で比率判定
    - どちらもない → デフォルト配分
    """
    text = str(text or "")

    # ★ 行ごとに分割
    lines = text.splitlines()

    # ★ プラン名行 + その次の行を除外
    skip_indices = set()
    for i, ln in enumerate(lines):
        if "プラン名" in ln:
            skip_indices.add(i)
            if i + 1 < len(lines):
                skip_indices.add(i + 1)

    answer_lines = [
        ln for i, ln in enumerate(lines)
        if i not in skip_indices
    ]

    # 回答テキストを結合
    answer_text = "\n".join(answer_lines)

    pasta_word = "パスタ"
    pizza_word = "ピザ"

    # パスタ・ピザの記述なし → デフォルト
    if pasta_word not in answer_text and pizza_word not in answer_text:
        p, z = _default_main_distribution(guest_count)
        return {"パスタ": p, "ピザ": z}

    # 出現回数カウント
    pasta_occ = answer_text.count(pasta_word)
    pizza_occ = answer_text.count(pizza_word)

    # ★ 一方だけ出現する → 全員そのメイン
    if pizza_occ > 0 and pasta_occ == 0:
        return {"パスタ": 0, "ピザ": guest_count}
    if pasta_occ > 0 and pizza_occ == 0:
        return {"パスタ": guest_count, "ピザ": 0}

    # 両方出現する場合は比率で分配
    total_occ = pasta_occ + pizza_occ
    if total_occ <= 0:
        p, z = _default_main_distribution(guest_count)
        return {"パスタ": p, "ピザ": z}

    # 出現回数 = 人数 の場合はそのまま
    if total_occ == guest_count:
        return {"パスタ": pasta_occ, "ピザ": pizza_occ}

    # 比率で補正
    pasta_num = round(guest_count * pasta_occ / total_occ)
    pizza_num = guest_count - pasta_num

    return {
        "パスタ": max(pasta_num, 0),
        "ピザ": max(pizza_num, 0),
    }




# =========================
# コース・予約系ヘルパー
# =========================
def course_has_main_item(course_id: str) -> bool:
    res = (
        supabase.table("course_items")
        .select("id, item_name")
        .eq("course_id", course_id)
        .execute()
    )
    rows = res.data or []
    return any(r["item_name"] == "メイン" for r in rows)


def fetch_courses():
    # 予約で選べるのは「有効なコース」のみにする
    res = (
        supabase.table("course_master")
        .select("*")
        .eq("is_active", True)
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


def fetch_course_items(course_id):
    res = (
        supabase.table("course_items")
        .select("id, display_order, offset_minutes, item_name")
        .eq("course_id", course_id)
        .order("display_order", desc=False)
        .execute()
    )
    return res.data or []


# 予約時間の選択肢（固定）
TIME_OPTIONS = ["18:00", "18:30", "20:30", "21:00"]

# 時間帯ごとのバッティングルール
TIME_CONFLICT_RULES = {
    "18:00": {"18:00", "18:30"},
    "18:30": {"18:00", "18:30", "20:30"},
    "20:30": {"18:30", "20:30", "21:00"},
    "21:00": {"20:30", "21:00"},
}


def is_slot_conflicted(reserved_at: datetime, table_no: str, exclude_reservation_id: str = None) -> bool:
    """
    指定された reserved_at / table_no の組み合わせが予約ルール違反かどうかを判定する。

    ルール:
      - 同じテーブルに18:00の予約がある → 18:00, 18:30 はNG
      - 同じテーブルに18:30の予約がある → 18:00, 18:30, 20:30 はNG
      - 同じテーブルに20:30の予約がある → 18:30, 20:30, 21:00 はNG
      - 同じテーブルに21:00の予約がある → 20:30, 21:00 はNG

    status = 'cancelled' は空きとみなす。
    exclude_reservation_id が指定されている場合、その予約IDは除外して判定。
    """
    # 対象日の0:00〜24:00の範囲で、そのテーブルの予約を取得
    target_date = reserved_at.date()
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = start_dt + timedelta(days=1)

    target_time_str = reserved_at.strftime("%H:%M")
    conflict_times = TIME_CONFLICT_RULES.get(target_time_str, {target_time_str})

    query = (
        supabase.table("course_reservations")
        .select("id, reserved_at, status")
        .gte("reserved_at", start_dt.isoformat())
        .lt("reserved_at", end_dt.isoformat())
        .eq("table_no", table_no)
        .neq("status", "cancelled")
    )
    if exclude_reservation_id:
        query = query.neq("id", exclude_reservation_id)

    res = query.execute()
    rows = res.data or []

    for row in rows:
        existing_time_str = datetime.fromisoformat(row["reserved_at"]).strftime("%H:%M")
        # 既存予約の時間が、この時間帯のNGリストに含まれていればバッティング
        if existing_time_str in conflict_times:
            return True

    return False


def create_reservation_and_progress(
    course_id,
    reserved_at,
    guest_name,
    guest_count,
    table_no,
    note,
    main_choice=None,
    main_detail_counts=None,  # ★ 追加: {"パスタ": 1, "ピザ": 1} みたいな dict
    is_birthday=False,
):
    # 1. 同じ時間・同じテーブルに予約がないか確認
    if is_slot_conflicted(reserved_at, table_no):
        return False, "この時間帯は同じテーブルに別の予約が入っているため、登録できません。"

    # 2. 予約を登録
    reservation_data = {
        "course_id": course_id,
        "reserved_at": reserved_at.isoformat(),
        "guest_name": guest_name,           # 必須
        "guest_count": guest_count,
        "table_no": table_no,               # 必須（プルダウン）
        "status": "reserved",
        "note": note or None,
        "main_choice": main_choice,
        "is_birthday": is_birthday,
    }
    try:
        res = supabase.table("course_reservations").insert(reservation_data).execute()
        reservation = res.data[0]
        reservation_id = reservation["id"]
    except Exception as e:
        return False, f"予約登録に失敗しました: {e}"

    # 3. コースアイテム取得
    items = fetch_course_items(course_id)
    if not items:
        # 予約は作れたが、コースに商品が無い場合
        return True, "予約は登録しましたが、このコースには商品が登録されていません。"

    # 4. course_progress をまとめて作成
    progress_rows = []
    for item in items:
        scheduled_time = reserved_at + timedelta(minutes=int(item["offset_minutes"]))

        if item["item_name"] == "メイン" and main_detail_counts:
            # ★ メインは種類ごとに別レコードで作成
            for name, cnt in main_detail_counts.items():
                if cnt <= 0:
                    continue
                progress_rows.append(
                    {
                        "reservation_id": reservation_id,
                        "course_item_id": item["id"],
                        "scheduled_time": scheduled_time.isoformat(),
                        "is_cooked": False,
                        "is_served": False,
                        "main_detail": name,       # パスタ / ピザ
                        "quantity": int(cnt),      # 皿数
                    }
                )
        else:
            # メイン以外は従来どおり 1レコード
            progress_rows.append(
                {
                    "reservation_id": reservation_id,
                    "course_item_id": item["id"],
                    "scheduled_time": scheduled_time.isoformat(),
                    "is_cooked": False,
                    "is_served": False,
                    "quantity": 1,              # 新カラム（デフォルト1）
                }
            )

    try:
        supabase.table("course_progress").insert(progress_rows).execute()
    except Exception as e:
        return False, f"予約は登録しましたが、進行テーブルの作成に失敗しました: {e}"

    return True, "予約とコース進行を登録しました。"


def fetch_reservations_for_date(target_date: date):
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0, 0))

    res = (
        supabase.table("course_reservations")
        .select("id, reserved_at, guest_name, guest_count, table_no, status, note, course_id, main_choice, is_birthday")
        .gte("reserved_at", start_dt.isoformat())
        .lt("reserved_at", end_dt.isoformat())
        .order("reserved_at", desc=False)  # 時間でざっくりソート
        .execute()
    )
    rows = res.data or []

    # 時間 → テーブル順に並べ替え（Python側）
    def sort_key(r):
        dt = datetime.fromisoformat(r["reserved_at"])
        table = r.get("table_no") or ""
        table_idx = TABLE_ORDER.get(table, 999)  # 想定外テーブルは末尾へ
        return (dt, table_idx)

    rows.sort(key=sort_key)
    return rows


def update_reservation_basic(
    reservation_id: str,
    guest_name: str,
    guest_count: int,
    table_no: str,
    status: str,
    note: str,
    reserved_at: datetime,
    main_choice: Optional[str],
    is_birthday: bool,
):
    """
    予約の基本情報を更新（コースと日時は今回は編集対象外）。
    同じ時間×テーブルの重複チェックも行う。
    メイン人数が変更された場合は、メインの商品進行も作り直す。
    """
    # 同じ時間・テーブルの重複チェック（自分自身は除外）
    if is_slot_conflicted(reserved_at, table_no, exclude_reservation_id=reservation_id):
        return False, "この時間帯は同じテーブルに別の予約が入っているため、変更できません。"

    update_data = {
        "guest_name": guest_name,
        "guest_count": guest_count,
        "table_no": table_no,
        "status": status,
        "note": note or None,
        "main_choice": main_choice,
        "is_birthday": is_birthday,
    }

    try:
        supabase.table("course_reservations").update(update_data).eq("id", reservation_id).execute()
    except Exception as e:
        return False, f"予約情報の更新に失敗しました: {e}"

    # ★ メインの内訳を course_progress にも反映
    try:
        # この予約の course_id を取得
        res = (
            supabase.table("course_reservations")
            .select("course_id")
            .eq("id", reservation_id)
            .single()
            .execute()
        )
        course_id = res.data["course_id"]

        if course_has_main_item(course_id):
            # メイン用の course_item 一覧
            res_items = (
                supabase.table("course_items")
                .select("id, offset_minutes, item_name")
                .eq("course_id", course_id)
                .eq("item_name", "メイン")
                .execute()
            )
            main_items = res_items.data or []
            main_item_ids = [i["id"] for i in main_items]

            if main_item_ids:
                # 既存のメイン progress を削除
                supabase.table("course_progress").delete() \
                    .eq("reservation_id", reservation_id) \
                    .in_("course_item_id", main_item_ids) \
                    .execute()

                # main_choice から人数 dict を再生成
                counts = parse_main_choice_to_counts(main_choice) if main_choice else {}

                if counts:
                    progress_rows = []
                    for item in main_items:
                        scheduled_time = reserved_at + timedelta(minutes=int(item["offset_minutes"]))
                        for name, cnt in counts.items():
                            if cnt <= 0:
                                continue
                            progress_rows.append(
                                {
                                    "reservation_id": reservation_id,
                                    "course_item_id": item["id"],
                                    "scheduled_time": scheduled_time.isoformat(),
                                    "is_cooked": False,
                                    "is_served": False,
                                    "main_detail": name,
                                    "quantity": int(cnt),
                                }
                            )
                    if progress_rows:
                        supabase.table("course_progress").insert(progress_rows).execute()

    except Exception as e:
        # 予約自体は更新できているので、ここは警告に留める
        st.warning(f"メイン料理の進行データ更新に失敗しました: {e}")

    return True, "予約情報を更新しました。"


def delete_reservation(reservation_id: str):
    """
    予約の削除。
    course_progress 側に ON DELETE CASCADE が付いていればそれでOK。
    そうでない場合は、先に course_progress を削除する必要がある。
    ここではとりあえず reservations からの削除を試みる。
    """
    try:
        supabase.table("course_reservations").delete().eq("id", reservation_id).execute()
        return True, "予約を削除しました。"
    except Exception as e:
        return False, f"予約の削除に失敗しました: {e}"


def delete_reservations_for_date(target_date: date):
    """
    指定日の予約を一括削除する。
    course_progress 側に ON DELETE CASCADE が付いていれば連動して削除される。
    """
    start_dt = datetime.combine(target_date, time(0, 0, 0))
    end_dt = start_dt + timedelta(days=1)

    try:
        (
            supabase.table("course_reservations")
            .delete()
            .gte("reserved_at", start_dt.isoformat())
            .lt("reserved_at", end_dt.isoformat())
            .execute()
        )
        return True, f"{target_date.strftime('%Y/%m/%d')} の予約をすべて削除しました。"
    except Exception as e:
        return False, f"{target_date.strftime('%Y/%m/%d')} の一括削除に失敗しました: {e}"


# =========================
# Excel 解析ヘルパー
# =========================
def _parse_reservation_date_from_df(df: pd.DataFrame) -> date:
    """
    ExcelのA2などから予約日を取得。
    '2025/12/01 (月)' のような文字列から日付部分だけを抜く。
    見つからなければ get_today_jst() を返す。
    """
    reserve_date = get_today_jst()
    try:
        # 2行目あたりをざっくり見る
        for col in range(min(5, df.shape[1])):
            v = df.iat[1, col] if df.shape[0] > 1 else None
            if isinstance(v, str):
                m = re.search(r"(\d{4}/\d{1,2}/\d{1,2})", v)
                if m:
                    reserve_date = datetime.strptime(m.group(1), "%Y/%m/%d").date()
                    break
    except Exception:
        pass
    return reserve_date


def _find_time_rows(df: pd.DataFrame):
    """
    A列の中から '18:00' などの時間が入っている行インデックスを取得。
    """
    time_rows = []
    for idx, v in df[0].items():
        if isinstance(v, str) and re.match(r"\d{1,2}:\d{2}", v.strip()):
            time_rows.append(idx)
    return time_rows


def _detect_table_no(block_df: pd.DataFrame) -> Optional[str]:
    """
    ブロック内のD列からテーブル番号を推定。
    '1-T5本棚' のようなケースもあるので、TABLE_OPTIONSの部分一致で最初に見つかったものを採用。
    """
    for _, row in block_df.iterrows():
        val = row[3] if 3 in row else None
        if not isinstance(val, str):
            continue
        s = val.strip()
        if not s:
            continue
        for t in TABLE_OPTIONS:
            if t in s:
                return t
    return None


def _build_block_text(block_df: pd.DataFrame) -> str:
    """
    ブロック内の「コース」「備考」相当（E,F,G列 = index 4,5,6）を結合したテキスト。
    """
    texts = []
    for _, row in block_df.iterrows():
        for col in [4, 5, 6]:
            if col in row:
                v = row[col]
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
    return "\n".join(texts)


def _normalize_text_for_course(s: str) -> str:
    """
    コース名マッチ用にテキストを正規化：
    ・改行
    ・スペース（半角・全角）
    を削除する
    """
    return (
        str(s or "")
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "")
        .replace("　", "")
    )


def _longest_common_substring(a: str, b: str) -> int:
    """
    a, b の間で「連続して一致している部分文字列」の最大長を返す。
    例: a='クリスマスディナー', b='クリスマスツリー' → 'クリスマス' で 5
    """
    max_len = 0
    len_a, len_b = len(a), len(b)

    for i in range(len_a):
        for j in range(len_b):
            l = 0
            while i + l < len_a and j + l < len_b and a[i + l] == b[j + l]:
                l += 1
            if l > max_len:
                max_len = l
    return max_len


def _detect_course_from_text(block_text: str, courses) -> Tuple[Optional[str], Optional[str]]:
    """
    ブロック内テキストとアプリに登録されているコース名を部分一致で紐づけ。
    改行・空白を除去した上で、
    - コース名全体が含まれていれば、その長さをスコア
    - そうでなければ「連続して一致している部分」の最大長をスコア
    最長スコアのコースを採用するが、スコアが6未満ならノーヒット扱いにする。
      → 「クリスマス」(5文字) だけの一致では拾わない
      → 「クリスマスディナー」など十分長い一致があれば拾う
    """
    norm = _normalize_text_for_course(block_text)

    best_course = None
    best_score = 0

    for c in courses:
        name = c.get("name") or ""
        if not name:
            continue

        name_norm = _normalize_text_for_course(name)
        if not name_norm:
            continue

        # 1) 片方がもう片方を完全に含んでいる場合
        if name_norm in norm or norm in name_norm:
            score = min(len(name_norm), len(norm))
        else:
            # 2) 最長共通部分文字列の長さをスコアにする
            score = _longest_common_substring(name_norm, norm)

        if score > best_score:
            best_score = score
            best_course = c

    # ★ スコア6未満はノーヒット扱い（「クリスマス」単体などの誤検出を防ぐ）
    if best_course and best_score >= 6:
        return best_course["id"], best_course["name"]

    return None, None






def _detect_is_birthday(block_text: str) -> bool:
    """
    6・7列目の「備考」にクリスマスディナーコースが含まれている、かつ、
    「バースデー」もしくは「プレート」が含まれている場合にTrue。
    改行・スペースを除去した正規化テキストで判定する。
    """
    norm = _normalize_text_for_course(block_text)

    if "クリスマスディナーコース" in norm and (
        "バースデー" in norm or "プレート" in norm
    ):
        return True
    return False




def parse_excel_reservations(excel_file, courses):
    """
    Restyの予約Excelからコース予約候補を抽出する。

    戻り値: list[dict]
        {
            "reserved_at": datetime,
            "guest_name": str,
            "guest_count": int,
            "table_no": Optional[str],
            "course_id": Optional[str],
            "course_name": Optional[str],
            "is_birthday": bool,
            "main_counts": Optional[dict],
            "main_choice": Optional[str],
            "block_text": str,
        }
    """
    df = pd.read_excel(excel_file, header=None)
    reserve_date = _parse_reservation_date_from_df(df)
    time_rows = _find_time_rows(df)

    results = []

    if not time_rows:
        return results

    # この日の有効コースのうち、「メイン」を持つコースを事前にメモ
    main_course_ids = set()
    for c in courses:
        cid = c["id"]
        try:
            if course_has_main_item(cid):
                main_course_ids.add(cid)
        except Exception:
            pass

    for i, start in enumerate(time_rows):
        end = time_rows[i + 1] - 1 if i + 1 < len(time_rows) else df.index[-1]
        block_df = df.iloc[start : end + 1, :]

        # 予約時間
        time_raw = str(block_df.iat[0, 0]).strip()
        mt = re.search(r"(\d{1,2}):(\d{2})", time_raw)
        if not mt:
            continue
        hour, minute = int(mt.group(1)), int(mt.group(2))
        reserved_time = time(hour, minute)
        reserved_at = datetime.combine(reserve_date, reserved_time)

        # お名前
        guest_name = ""
        val_name = block_df.iat[0, 1]
        if isinstance(val_name, str):
            guest_name = val_name.strip()

        # 人数
        guest_count = 1
        val_guest = block_df.iat[0, 2]
        if isinstance(val_guest, str):
            mg = re.search(r"\d+", val_guest)
            if mg:
                guest_count = int(mg.group())

        # テーブル番号
        table_no = _detect_table_no(block_df)

        # コース・備考テキスト
        block_text = _build_block_text(block_df)

        # コース判定
        course_id, course_name = _detect_course_from_text(block_text, courses)

        # バースデーフラグ
        is_birthday = _detect_is_birthday(block_text)

        # メイン人数
        main_counts = None
        main_choice = None
        if course_id and course_id in main_course_ids:
            main_counts = parse_main_counts_from_text(guest_count, block_text)
            main_choice = counts_to_main_choice(main_counts)

        results.append(
            {
                "reserved_at": reserved_at,
                "guest_name": guest_name,
                "guest_count": guest_count,
                "table_no": table_no,
                "course_id": course_id,
                "course_name": course_name,
                "is_birthday": is_birthday,
                "main_counts": main_counts,
                "main_choice": main_choice,
                "block_text": block_text,
            }
        )

    return results


# =========================
# メイン画面
# =========================
def show():
    st.subheader("コース予約登録")

    # 前回成功時のメッセージを表示（1回だけ）
    success_msg = st.session_state.pop("reservation_success_message", None)
    if success_msg:
        st.success(success_msg)

    courses = fetch_courses()
    if not courses:
        st.warning("有効なコースがまだ登録されていません。先に『コースマスタ管理』でコースを登録・有効化してください。")
        return

    # フォームの「バージョン」：成功時だけ +1 して全ウィジェットの key を変える
    form_version = st.session_state.get("reservation_form_version", 0)

    # ======================
    # 予約登録フォーム
    # ======================
    st.markdown("### 新規予約")

    form_key_suffix = f"_v{form_version}"

    # エラー時は入力を残したいので clear_on_submit=False
    with st.form(f"reservation_form{form_key_suffix}", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            # コース選択
            selected_course_name = st.selectbox(
                "コースを選択",
                [c["name"] for c in courses],
                key=f"course_select{form_key_suffix}",
            )

            # ここで course を特定
            course_for_form = next(c for c in courses if c["name"] == selected_course_name)
            selected_course_id = course_for_form["id"]
            has_main = course_has_main_item(selected_course_id)

            guest_name = st.text_input(
                "お名前（必須）",
                key=f"guest_name_input{form_key_suffix}",
            )

            guest_count = st.number_input(
                "人数",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                key=f"guest_count_input{form_key_suffix}",
            )

            # ★ バースデー利用フラグ
            is_birthday = st.checkbox(
                "バースデー利用",
                value=False,
                key=f"is_birthday_input{form_key_suffix}",
            )

            # テーブル番号はプルダウン（必須）
            table_select_options = ["テーブルを選択してください"] + TABLE_OPTIONS
            table_selected = st.selectbox(
                "テーブル番号（必須）",
                options=table_select_options,
                index=0,
                key=f"table_no_input{form_key_suffix}",
            )

        with col2:
            date_input_val = st.date_input(
                "予約日",
                value=get_today_jst(),
                key=f"reservation_date{form_key_suffix}",
            )

            time_str = st.selectbox(
                "予約時間",
                TIME_OPTIONS,
                key=f"reservation_time{form_key_suffix}",
            )
            time_input_val = datetime.strptime(time_str, "%H:%M").time()

            note = st.text_area(
                "メモ（任意）",
                key=f"reservation_note{form_key_suffix}",
            )

        # メイン料理の人数入力
        main_counts = {}

        if has_main:
            st.markdown("#### メイン料理の内訳")

            for name in MAIN_OPTIONS:
                main_counts[name] = st.number_input(
                    f"{name} の人数",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                    key=f"main_{name}{form_key_suffix}",
                )

        submitted = st.form_submit_button("予約を登録")
        if submitted:
            if not guest_name.strip():
                st.warning("お名前を入力してください。")
            elif table_selected == "テーブルを選択してください":
                st.warning("テーブル番号を選択してください。")
            else:
                reserved_at = datetime.combine(date_input_val, time_input_val)

                # メインがあるコースなら、人数チェックと文字列生成
                main_choice_str = None
                if has_main:
                    total_main = sum(main_counts.values())
                    guest_count_int = int(guest_count)

                    if total_main != guest_count_int:
                        st.warning(
                            f"メイン料理の人数合計（{total_main}名）が"
                            f"予約人数（{guest_count_int}名）と一致していません。"
                        )
                        st.stop()

                    parts = [
                        f"{name}：{cnt}"
                        for name, cnt in main_counts.items()
                        if cnt > 0
                    ]
                    main_choice_str = "、".join(parts) if parts else None

                ok, msg = create_reservation_and_progress(
                    course_id=course_for_form["id"],
                    reserved_at=reserved_at,
                    guest_name=guest_name.strip(),
                    guest_count=int(guest_count),
                    table_no=table_selected,
                    note=note.strip() or None,
                    main_choice=main_choice_str,              # 文字列
                    main_detail_counts=main_counts if has_main else None,  # ★ 追加
                    is_birthday=is_birthday,
                )

                if ok:
                    # ★ 成功したらバージョンを +1 → 次の描画で key が全部変わるのでフォームが初期化される
                    st.session_state["reservation_form_version"] = form_version + 1
                    st.session_state["reservation_success_message"] = msg
                    st.rerun()
                else:
                    # 失敗時は入力を残したままエラー表示
                    st.error(msg)

    # ======================
    # Excelから予約取り込み
    # ======================
    st.markdown("### Excelから予約を取り込む")

    excel_file = st.file_uploader(
        "予約エクセルファイル（Resty出力 .xlsx）",
        type=["xlsx"],
        key="reservation_excel_uploader",
    )

    parsed_from_excel = []
    if excel_file is not None:
        try:
            parsed_from_excel = parse_excel_reservations(excel_file, courses)
        except Exception as e:
            st.error(f"Excelの解析に失敗しました: {e}")

    if parsed_from_excel:
        total_count_excel = len(parsed_from_excel)
        course_count_excel = sum(1 for r in parsed_from_excel if r["course_id"])

        st.caption(
            f"検出された予約件数: {total_count_excel}件 / "
            f"このうちコースと紐付けできた予約: {course_count_excel}件"
        )

        # プレビュー表示
        preview_rows = []
        for r in parsed_from_excel:
            preview_rows.append(
                {
                    "時間": r["reserved_at"].strftime("%H:%M"),
                    "お名前": r["guest_name"],
                    "人数": r["guest_count"],
                    "テーブル": r["table_no"] or "",
                    "コース": r["course_name"] or "",
                    "バースデー": "○" if r["is_birthday"] else "",
                    "メイン内訳": r["main_choice"] or "",
                }
            )
        st.dataframe(pd.DataFrame(preview_rows))

        if course_count_excel > 0:
            if st.button("この内容でコース予約を一括登録する", key="btn_register_excel"):
                success_count = 0
                warn_messages = []

                for r in parsed_from_excel:
                    # コースと紐づいている予約のみ登録
                    if not r["course_id"]:
                        continue

                    if not r["table_no"]:
                        warn_messages.append(
                            f"{r['guest_name']}様（{r['reserved_at'].strftime('%H:%M')}）: "
                            f"テーブル番号が判定できなかったためスキップしました。"
                        )
                        continue

                    ok, msg = create_reservation_and_progress(
                        course_id=r["course_id"],
                        reserved_at=r["reserved_at"],
                        guest_name=r["guest_name"],
                        guest_count=r["guest_count"],
                        table_no=r["table_no"],
                        note=r["block_text"],
                        main_choice=r["main_choice"],
                        main_detail_counts=r["main_counts"],
                        is_birthday=r["is_birthday"],
                    )
                    if ok:
                        success_count += 1
                    else:
                        warn_messages.append(
                            f"{r['guest_name']}様（{r['reserved_at'].strftime('%H:%M')}）: {msg}"
                        )

                if success_count > 0:
                    st.session_state["reservation_success_message"] = (
                        f"Excelからコース予約を {success_count}件 登録しました。"
                    )
                    st.rerun()
                else:
                    st.info("登録できた予約はありませんでした。")

                for m in warn_messages:
                    st.warning(m)

    # ======================
    # 予約一覧（任意の日付）＋ 編集・削除
    # ======================
    st.markdown("### 予約一覧（対象日を選択）")

    list_date = st.date_input("予約一覧の対象日", value=get_today_jst(), key="list_date")
    reservations = fetch_reservations_for_date(list_date)

    if not reservations:
        st.info("該当日の予約は登録されていません。")
        return

    # course_id → name のマップ（表示用）
    course_map = {c["id"]: c["name"] for c in courses}

    # -------------------------
    # 予約件数 + 時間帯別件数 + コース別件数
    # -------------------------
    total_count = len(reservations)

    # 時間帯ごとにカウント
    time_counter = Counter()
    for r in reservations:
        dt = datetime.fromisoformat(r["reserved_at"])
        time_str = dt.strftime("%H:%M")
        if time_str in TIME_OPTIONS:
            time_counter[time_str] += 1

    # メイン表示（総件数）
    st.markdown(
        f"#### {list_date.strftime('%Y/%m/%d')} の予約件数：**{total_count}件**"
    )

    # 時間帯ごとの件数表示（すべてのスロットを表示）
    time_lines = []
    for slot in TIME_OPTIONS:  # ["18:00", "18:30", "20:30", "21:00"]
        count = time_counter.get(slot, 0)
        time_lines.append(f"{slot}: {count}件")
    st.caption("時間帯別：" + " / ".join(time_lines))

    # コース別件数
    course_counter = Counter()
    for r in reservations:
        cid = r.get("course_id")
        if cid:
            course_counter[cid] += 1

    course_lines = []
    for c in courses:
        cid = c["id"]
        cnt = course_counter.get(cid, 0)
        if cnt > 0:
            course_lines.append(f"{c['name']}: {cnt}件")

    if course_lines:
        st.caption("コース別：" + " / ".join(course_lines))

    # -------------------------
    # ★ この日の予約を一括削除
    # -------------------------
    with st.expander("この日の予約を一括削除する", expanded=False):
        st.warning(
            f"{list_date.strftime('%Y/%m/%d')} の予約がすべて削除されます。"
            "この操作は取り消せません。"
        )
        confirm_delete_all = st.checkbox(
            "本当にこの日の全予約を削除する",
            key=f"confirm_delete_all_{list_date.strftime('%Y%m%d')}",
        )
        delete_all_btn = st.button(
            "この日の予約をすべて削除する",
            key=f"delete_all_btn_{list_date.strftime('%Y%m%d')}",
            disabled=not confirm_delete_all,
        )

        if delete_all_btn:
            ok, msg = delete_reservations_for_date(list_date)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # -------------------------
    # 個別の予約 編集・削除
    # -------------------------
    for r in reservations:
        res_time = datetime.fromisoformat(r["reserved_at"])
        main_label = ""
        if r.get("main_choice"):
            main_label = f" / メイン: {r['main_choice']}"

        title = (
            f"{res_time.strftime('%Y-%m-%d %H:%M')} / "
            f"{(r['guest_name'] or 'お名前未入力')} 様 / "
            f"テーブル: {r['table_no'] or '-'} / "
            f"コース: {course_map.get(r['course_id'], '不明')}"
            f"{main_label}"
        )

        with st.expander(title):
            with st.form(f"edit_reservation_form_{r['id']}"):
                c1, c2 = st.columns(2)

                with c1:
                    guest_name_edit = st.text_input(
                        "お名前（必須）",
                        value=r.get("guest_name") or "",
                        key=f"name_{r['id']}",
                    )

                    guest_count_edit = st.number_input(
                        "人数",
                        min_value=1,
                        max_value=20,
                        value=int(r.get("guest_count") or 1),
                        step=1,
                        key=f"count_{r['id']}",
                    )

                    # テーブル番号（必須、プルダウン）
                    table_current = r.get("table_no") or ""
                    if table_current in TABLE_OPTIONS:
                        table_index = TABLE_OPTIONS.index(table_current)
                    else:
                        table_index = 0
                    table_no_edit = st.selectbox(
                        "テーブル番号（必須）",
                        options=TABLE_OPTIONS,
                        index=table_index,
                        key=f"table_{r['id']}",
                    )

                with c2:
                    # ステータス編集
                    status_options = ["reserved", "arrived", "cancelled", "completed"]
                    status_current = r.get("status") or "reserved"
                    if status_current in status_options:
                        status_index = status_options.index(status_current)
                    else:
                        status_index = 0
                    status_edit = st.selectbox(
                        "ステータス",
                        options=status_options,
                        index=status_index,
                        key=f"status_{r['id']}",
                    )

                    note_edit = st.text_area(
                        "メモ（任意）",
                        value=r.get("note") or "",
                        key=f"note_{r['id']}",
                    )

                    # ★ バースデー利用フラグ（編集用）
                    is_birthday_current = bool(r.get("is_birthday"))
                    is_birthday_edit = st.checkbox(
                        "バースデー利用",
                        value=is_birthday_current,
                        key=f"is_birthday_{r['id']}",
                    )

                    # この予約のコースに「メイン」アイテムがあるかどうか
                    has_main_for_row = course_has_main_item(r["course_id"])

                    # メイン人数（編集用）
                    main_counts_edit = None
                    if has_main_for_row:
                        st.markdown("メイン料理の内訳（編集）")

                        # 既存の main_choice を人数dictにパース
                        main_counts_edit = parse_main_choice_to_counts(r.get("main_choice"))

                        for name in MAIN_OPTIONS:
                            main_counts_edit[name] = st.number_input(
                                f"{name} の人数",
                                min_value=0,
                                max_value=20,
                                value=main_counts_edit.get(name, 0),
                                step=1,
                                key=f"edit_main_{name}_{r['id']}",
                            )

                    # 予約日時は今回は編集不可（必要なら後で実装）
                    st.caption(f"予約日時（変更不可）: {res_time.strftime('%Y-%m-%d %H:%M')}")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    update_btn = st.form_submit_button("この予約を更新")
                with col_btn2:
                    delete_btn = st.form_submit_button("この予約を削除")

                # 更新処理
                if update_btn:
                    if not guest_name_edit.strip():
                        st.warning("お名前を入力してください。")
                    elif not table_no_edit:
                        st.warning("テーブル番号を選択してください。")
                    else:
                        # メイン人数のチェックと main_choice の再生成
                        main_choice_edit = None
                        if has_main_for_row:
                            # main_counts_edit は上で number_input を通して更新済み
                            total_main_edit = sum(main_counts_edit.values())
                            guest_count_int_edit = int(guest_count_edit)

                            if total_main_edit != guest_count_int_edit:
                                st.warning(
                                    f"メイン料理の人数合計（{total_main_edit}名）が "
                                    f"人数（{guest_count_int_edit}名）と一致していません。"
                                )
                                st.stop()

                            main_choice_edit = counts_to_main_choice(main_counts_edit)

                        ok, msg = update_reservation_basic(
                            reservation_id=r["id"],
                            guest_name=guest_name_edit.strip(),
                            guest_count=int(guest_count_edit),
                            table_no=table_no_edit,
                            status=status_edit,
                            note=note_edit.strip(),
                            reserved_at=res_time,
                            main_choice=main_choice_edit,
                            is_birthday=is_birthday_edit,
                        )
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                # 削除処理
                if delete_btn:
                    # 軽い確認（本格的な確認UIが必要ならチェックボックスを追加してもOK）
                    ok, msg = delete_reservation(r["id"])
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
