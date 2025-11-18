# modules/course_master.py

import streamlit as st
from .supabase_client import supabase


def fetch_courses():
    res = supabase.table("course_master").select("*").order("created_at", desc=False).execute()
    return res.data or []


def fetch_course_items(course_id):
    res = (
        supabase
        .table("course_items")
        .select("*")
        .eq("course_id", course_id)
        .order("display_order", desc=False)
        .execute()
    )
    return res.data or []


def show():
    st.subheader("コースマスタ管理")

    courses = fetch_courses()

    col_left, col_right = st.columns([1, 2])

    # ======================
    # 左：コース一覧 & 新規追加
    # ======================
    with col_left:
        st.markdown("### コース一覧")

        course_names = ["（新規コースを追加）"] + [c["name"] for c in courses]
        selected_name = st.selectbox("編集するコースを選択", course_names)

        st.markdown("### 新規コース追加")
        with st.form("add_course_form", clear_on_submit=True):
            new_course_name = st.text_input("コース名", "")
            new_course_desc = st.text_area("コース説明（任意）", "")
            is_active = st.checkbox("有効", value=True)

            submitted = st.form_submit_button("コースを追加")
            if submitted:
                if not new_course_name.strip():
                    st.warning("コース名を入力してください。")
                else:
                    data = {
                        "name": new_course_name.strip(),
                        "description": new_course_desc.strip() or None,
                        "is_active": is_active,
                    }
                    try:
                        supabase.table("course_master").insert(data).execute()
                        st.success("コースを追加しました。ページを再読み込みすると反映されます。")
                    except Exception as e:
                        st.error(f"コース追加に失敗しました: {e}")

    # ======================
    # 右：選択コースの詳細 & 状態変更 & 削除 & 商品一覧
    # ======================
    with col_right:
        if selected_name != "（新規コースを追加）":
            course = next((c for c in courses if c["name"] == selected_name), None)
        else:
            course = courses[0] if courses else None

        if not course:
            st.info("コースがまだ登録されていません。左側からコースを追加してください。")
            return

        st.markdown(f"### コース詳細：{course['name']}")
        if course.get("description"):
            st.caption(course["description"])
        st.caption(f"状態: {'有効' if course.get('is_active', True) else '無効'}")

        # ---- 有効/無効の切り替え ----
        with st.form(f"course_active_form_{course['id']}"):
            is_active_new = st.checkbox(
                "このコースを有効にする",
                value=course.get("is_active", True),
            )
            update_active_btn = st.form_submit_button("状態を更新")

            if update_active_btn:
                try:
                    supabase.table("course_master").update(
                        {"is_active": is_active_new}
                    ).eq("id", course["id"]).execute()
                    st.success("コースの有効/無効状態を更新しました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"状態の更新に失敗しました: {e}")

        # ---- コース削除 ----
        st.markdown("#### コースを削除")
        with st.form(f"delete_course_form_{course['id']}"):
            st.warning(
                "このコースを削除すると、このコースに紐づく商品も削除されます。\n"
                "※ すでに予約が登録されている場合は、外部キー制約により削除できないことがあります。"
            )
            confirm_delete = st.checkbox("本当にこのコースを削除する", value=False)
            delete_btn = st.form_submit_button("コースを削除")

            if delete_btn:
                if not confirm_delete:
                    st.warning("削除する場合はチェックボックスにチェックを入れてください。")
                else:
                    try:
                        supabase.table("course_master").delete().eq("id", course["id"]).execute()
                        st.success("コースを削除しました。")
                        st.rerun()
                    except Exception as e:
                        st.error(
                            "コースの削除に失敗しました。\n"
                            "このコースに紐づく予約（course_reservations）が残っている可能性があります。\n"
                            f"詳細: {e}"
                        )

        st.markdown("---")

        # ---- 商品一覧（編集・削除付き）----
        items = fetch_course_items(course["id"])

        st.markdown("#### 商品一覧（編集・削除）")
        if items:
            for item in items:
                st.markdown("---")
                with st.form(f"edit_item_form_{item['id']}"):
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])

                    with c1:
                        item_name = st.text_input(
                            "商品名",
                            value=item["item_name"],
                            key=f"item_name_{item['id']}",
                        )

                    with c2:
                        offset_minutes = st.number_input(
                            "提供までの分数",
                            min_value=0,
                            max_value=600,
                            value=int(item["offset_minutes"]),
                            step=1,
                            key=f"offset_{item['id']}",
                        )

                    with c3:
                        display_order = st.number_input(
                            "表示順",
                            min_value=1,
                            max_value=200,
                            value=int(item["display_order"]),
                            step=1,
                            key=f"order_{item['id']}",
                        )

                    with c4:
                        st.write("")  # 余白
                        update_btn = st.form_submit_button("更新")
                        delete_btn = st.form_submit_button("削除")

                    # 更新処理
                    if update_btn:
                        if not item_name.strip():
                            st.warning("商品名を入力してください。")
                        else:
                            update_data = {
                                "item_name": item_name.strip(),
                                "offset_minutes": int(offset_minutes),
                                "display_order": int(display_order),
                            }
                            try:
                                supabase.table("course_items").update(update_data).eq("id", item["id"]).execute()
                                st.success("商品を更新しました。")
                                st.rerun()
                            except Exception as e:
                                st.error(f"商品の更新に失敗しました: {e}")

                    # 削除処理
                    if delete_btn:
                        try:
                            # 1. 先に course_progress 側の関連レコードを削除
                            supabase.table("course_progress").delete().eq("course_item_id", item["id"]).execute()

                            # 2. 次に course_items のレコードを削除
                            supabase.table("course_items").delete().eq("id", item["id"]).execute()

                            st.success("商品を削除しました。（関連する進行データも削除されました）")
                            st.rerun()
                        except Exception as e:
                            st.error(f"商品の削除に失敗しました: {e}")

        else:
            st.info("このコースにはまだ商品が登録されていません。")

        # ---- 商品追加 ----
        st.markdown("#### 商品を追加")

        max_order = max([i["display_order"] for i in items], default=0)
        default_order = max_order + 1

        with st.form("add_item_form", clear_on_submit=True):
            item_name = st.text_input("商品名", "")
            offset_minutes = st.number_input(
                "提供までの分数（開始から何分後か）",
                min_value=0,
                max_value=600,
                value=5,
                step=1,
            )
            display_order = st.number_input(
                "表示順（1,2,3...）",
                min_value=1,
                max_value=200,
                value=default_order,
                step=1,
            )
            memo = st.text_input("メモ（任意）", "")

            submitted_item = st.form_submit_button("商品を追加")
            if submitted_item:
                if not item_name.strip():
                    st.warning("商品名を入力してください。")
                else:
                    data = {
                        "course_id": course["id"],
                        "display_order": int(display_order),
                        "item_name": item_name.strip(),
                        "offset_minutes": int(offset_minutes),
                        "memo": memo.strip() or None,
                    }
                    try:
                        supabase.table("course_items").insert(data).execute()
                        st.success("商品を追加しました。ページを再読み込みすると反映されます。")
                    except Exception as e:
                        st.error(f"商品追加に失敗しました: {e}")

        st.caption("※ コース自体の有効/無効・削除は上部のフォームから操作できます。")
