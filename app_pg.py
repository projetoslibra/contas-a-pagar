# app_pg.py — Contas a Pagar (Streamlit + PostgreSQL)
# ---------------------------------------------------
# Como rodar local/na nuvem:
# 1) pip install -r requirements.txt
# 2) Configure secrets TOML com as chaves [postgres]
# 3) streamlit run app_pg.py
# ---------------------------------------------------

from contextlib import contextmanager
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from io import StringIO
import uuid

import pandas as pd
import streamlit as st
import psycopg2
from psycopg2 import pool

# =============================
# Config & constantes
# =============================

EMPRESAS = [
    "Libra Agente Autônomo",
    "Libra Banco",
    "Libra Capital ADM",
    "Libra Consignado",
    "Libra Garantidora",
    "Libra Holding",
    "Libra Negócios",
    "Libra Securitizadora",
    "Libra Seguros",
    "Libra Soluções em Cobrança",
    "Libra Treinamentos",
]

BRL = st.column_config.NumberColumn("Valor (R$)", format="R$ %,.2f")
DATE_COL = st.column_config.DateColumn("Data", format="DD/MM/YYYY")


def force_rerun():
    """Força recarregamento da página após uma ação."""
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# =============================
# Conexão com Postgres (pool)
# =============================

@st.cache_resource(show_spinner=False)
def get_pool():
    cfg = st.secrets["postgres"]
    return pool.SimpleConnectionPool(
        minconn=1,
        maxconn=6,
        host=cfg["host"],
        port=cfg.get("port", 5432),
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        sslmode=cfg.get("sslmode", "require"),
    )


@contextmanager
def get_conn():
    _pool = get_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


# =============================
# Inicialização / migração DB
# =============================

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        # Tabela principal
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS boletos (
                id SERIAL PRIMARY KEY,
                data_registro DATE NOT NULL,
                empresa_pagadora TEXT NOT NULL,
                valor NUMERIC(14,2) NOT NULL,
                data_vencimento DATE NOT NULL,
                beneficiario TEXT NOT NULL,
                banco TEXT,
                setor TEXT,
                status TEXT NOT NULL DEFAULT 'A_PAGAR',  -- A_PAGAR | PAGO | CANCELADO
                data_pagamento DATE,                     -- null se não pago
                obs TEXT,
                arquivo BYTEA,                           -- PDF em bytes
                arquivo_nome TEXT                        -- nome amigável do arquivo
            );
            """
        )
        # Índices úteis
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boletos_vcto ON boletos(data_vencimento);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boletos_emp  ON boletos(empresa_pagadora);")
        conn.commit()
        cur.close()


# =============================
# CRUD
# =============================

def insert_boleto(payload: dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO boletos(
                data_registro, empresa_pagadora, valor, data_vencimento,
                beneficiario, banco, setor, status, data_pagamento, obs,
                arquivo, arquivo_nome
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                payload["data_registro"],
                payload["empresa_pagadora"],
                float(payload["valor"]),
                payload["data_vencimento"],
                payload.get("beneficiario", ""),
                payload.get("banco", ""),
                payload.get("setor", ""),
                payload.get("status", "A_PAGAR"),
                payload.get("data_pagamento"),
                payload.get("obs", ""),
                payload.get("arquivo_bytes"),
                payload.get("arquivo_nome"),
            ),
        )
        conn.commit()
        cur.close()


def fetch_boletos(filters: dict | None = None) -> pd.DataFrame:
    base_sql = """
        SELECT
            id, data_registro, empresa_pagadora, valor, data_vencimento,
            beneficiario, banco, setor, status, data_pagamento, obs, arquivo_nome
        FROM boletos
        WHERE 1=1
    """
    params = []

    if filters:
        if empresa_list := filters.get("empresa_list"):
            marks = ",".join(["%s"] * len(empresa_list))
            base_sql += f" AND empresa_pagadora IN ({marks})"
            params.extend(empresa_list)
        if beneficiario_list := filters.get("beneficiario_list"):
            marks = ",".join(["%s"] * len(beneficiario_list))
            base_sql += f" AND beneficiario IN ({marks})"
            params.extend(beneficiario_list)
        if setor_list := filters.get("setor_list"):
            marks = ",".join(["%s"] * len(setor_list))
            base_sql += f" AND setor IN ({marks})"
            params.extend(setor_list)
        if status_list := filters.get("status_list"):
            marks = ",".join(["%s"] * len(status_list))
            base_sql += f" AND status IN ({marks})"
            params.extend(status_list)
        if dt_ini := filters.get("dt_ini"):
            base_sql += " AND data_vencimento >= %s"
            params.append(dt_ini)
        if dt_fim := filters.get("dt_fim"):
            base_sql += " AND data_vencimento <= %s"
            params.append(dt_fim)

    with get_conn() as conn:
        df = pd.read_sql_query(base_sql, conn, params=params)

    if not df.empty:
        for col in ["data_registro", "data_vencimento", "data_pagamento"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_pdf(id_: int):
    """Busca o PDF (BYTEA) e o nome do arquivo para download."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT arquivo, arquivo_nome FROM boletos WHERE id = %s", (id_,))
        row = cur.fetchone()
        cur.close()
    if not row:
        return None, None
    arquivo_bytes, arquivo_nome = row
    if arquivo_bytes is None:
        return None, None
    
        # 🔧 Converte memoryview/bytearray/etc. para bytes
    if isinstance(arquivo_bytes, memoryview):
        arquivo_bytes = arquivo_bytes.tobytes()
    elif not isinstance(arquivo_bytes, (bytes, bytearray)):
        arquivo_bytes = bytes(arquivo_bytes)
    else:
        arquivo_bytes = bytes(arquivo_bytes)  # garante tipo bytes

    if not arquivo_nome:
        arquivo_nome = f"boleto_{id_}.pdf"
    return arquivo_bytes, arquivo_nome


def update_rows(df_updated: pd.DataFrame, edited_rows: list[int]):
    if not edited_rows:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        for i in edited_rows:
            row = df_updated.iloc[i]
            cur.execute(
                """
                UPDATE boletos SET
                    data_registro = %s, empresa_pagadora = %s, valor = %s,
                    data_vencimento = %s, beneficiario = %s, banco = %s, setor = %s,
                    status = %s, data_pagamento = %s, obs = %s
                WHERE id = %s
                """,
                (
                    row["data_registro"].strftime("%Y-%m-%d") if pd.notna(row["data_registro"]) else None,
                    row["empresa_pagadora"],
                    float(row["valor"]),
                    row["data_vencimento"].strftime("%Y-%m-%d") if pd.notna(row["data_vencimento"]) else None,
                    row.get("beneficiario", ""),
                    row.get("banco", ""),
                    row.get("setor", ""),
                    row.get("status", "A_PAGAR"),
                    row["data_pagamento"].strftime("%Y-%m-%d") if pd.notna(row["data_pagamento"]) else None,
                    row.get("obs", ""),
                    int(row["id"]),
                ),
            )
        conn.commit()
        cur.close()


def delete_by_ids(ids: list[int]):
    if not ids:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        marks = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM boletos WHERE id IN ({marks})", tuple(ids))
        conn.commit()
        cur.close()


# =============================
# UI — Streamlit
# =============================

st.set_page_config(page_title="Contas a Pagar", page_icon="💸", layout="wide")
init_db()

st.title("💸 Contas a Pagar — Grupo Empresarial")

page = st.sidebar.radio("Navegação", ["Adicionar Boleto", "Dashboard"], index=0)

# 🔄 botão de refresh manual na sidebar
if st.sidebar.button("🔄 Atualizar página"):
    force_rerun()

# Paleta/formatadores
BRL = st.column_config.NumberColumn("Valor (R$)", format="R$ %,.2f")
DATE_COL = st.column_config.DateColumn("Data", format="DD/MM/YYYY")

# =============================
# Página: Adicionar Boleto
# =============================
if page == "Adicionar Boleto":
    st.subheader("Adicionar novo boleto")
    with st.form("form_boleto", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            data_registro = st.date_input("Data de registro", value=date.today())
            # select padronizado
            empresa_pagadora = st.selectbox("Empresa pagadora", EMPRESAS, index=0)
            valor = st.number_input("Valor do boleto (R$)", min_value=0.0, step=0.01, format="%.2f")
        with col2:
            data_vencimento = st.date_input("Data de vencimento", value=date.today() + relativedelta(days=7))
            beneficiario = st.text_input("Beneficiário", placeholder="Ex: Fornecedor XYZ")
            banco = st.text_input("Banco", placeholder="Ex: Itaú")
        with col3:
            setor = st.text_input("Setor", placeholder="Ex: Administrativo")
            status = st.selectbox("Status", ["A_PAGAR", "PAGO", "CANCELADO"], index=0)
            data_pagamento = st.date_input("Data de pagamento (se pago)", value=None, format="DD/MM/YYYY")

        pdf = st.file_uploader("Anexar boleto (PDF)", type=["pdf"], accept_multiple_files=False)
        obs = st.text_area("Observações", placeholder="Notas adicionais…")

        submitted = st.form_submit_button("➕ Adicionar Boleto")
        if submitted:
            # Prepara bytes e nome do PDF (BYTEA)
            arquivo_bytes = None
            arquivo_nome = None
            if pdf is not None:
                arquivo_bytes = pdf.read()
                safe_benef = (beneficiario or "boleto").replace("/", "-").replace("\\", "-")
                arquivo_nome = f"{date.today().isoformat()}_{uuid.uuid4().hex[:8]}_{safe_benef}.pdf"

            payload = {
                "data_registro": data_registro.strftime("%Y-%m-%d"),
                "empresa_pagadora": empresa_pagadora.strip(),
                "valor": valor,
                "data_vencimento": data_vencimento.strftime("%Y-%m-%d"),
                "beneficiario": beneficiario.strip(),
                "banco": banco.strip(),
                "setor": setor.strip(),
                "status": status,
                "data_pagamento": data_pagamento.strftime("%Y-%m-%d") if data_pagamento else None,
                "obs": obs.strip(),
                "arquivo_bytes": arquivo_bytes,
                "arquivo_nome": arquivo_nome,
            }
            # validações simples
            if not payload["empresa_pagadora"]:
                st.error("Informe a *Empresa pagadora*.")
            elif payload["valor"] <= 0:
                st.error("O *Valor do boleto* deve ser maior que zero.")
            else:
                insert_boleto(payload)
                st.success("Boleto adicionado com sucesso!")
                force_rerun()

    st.info("Dica: você pode importar boletos em lote via CSV no painel *Dashboard* (seção *Importar CSV*).")

# =============================
# Página: Dashboard
# =============================
else:
    st.subheader("Dashboard e Controle")

    # --- Filtros (1ª camada: datas/status) ---
    with st.expander("🔎 Filtros", expanded=True):
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            dt_ini = st.date_input("Vencimento — início", value=date.today() - relativedelta(months=1))
        with colf2:
            dt_fim = st.date_input("Vencimento — fim", value=date.today() + relativedelta(months=1))
        with colf3:
            status_list = st.multiselect("Status", ["A_PAGAR", "PAGO", "CANCELADO"], default=["A_PAGAR", "PAGO"])

        df_all = fetch_boletos(
            {
                "dt_ini": dt_ini.strftime("%Y-%m-%d"),
                "dt_fim": dt_fim.strftime("%Y-%m-%d"),
                "status_list": status_list,
            }
        )

        # --- Filtros (2ª camada: empresa/beneficiário/setor) ---
        emp_opts = EMPRESAS  # padronizado
        ben_opts = sorted(df_all["beneficiario"].dropna().unique().tolist()) if not df_all.empty else []
        set_opts = sorted(df_all["setor"].dropna().unique().tolist()) if not df_all.empty else []

        colff1, colff2, colff3 = st.columns(3)
        with colff1:
            empresa_list = st.multiselect("Empresa(s)", emp_opts, default=emp_opts)
        with colff2:
            beneficiario_list = st.multiselect("Beneficiário(s)", ben_opts, default=ben_opts)
        with colff3:
            setor_list = st.multiselect("Setor(es)", set_opts, default=set_opts)

        # aplica filtros adicionais
        df = df_all.copy()
        if empresa_list:
            df = df[df["empresa_pagadora"].isin(empresa_list)]
        if beneficiario_list:
            df = df[df["beneficiario"].isin(beneficiario_list)]
        if setor_list:
            df = df[df["setor"].isin(setor_list)]

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    total_a_pagar = float(df.loc[df["status"] == "A_PAGAR", "valor"].sum()) if not df.empty else 0.0
    total_pago = float(df.loc[df["status"] == "PAGO", "valor"].sum()) if not df.empty else 0.0
    qtd_boletos = int(df.shape[0]) if not df.empty else 0
    prox_7 = float(
        df.loc[
            (df["status"] == "A_PAGAR") & (df["data_vencimento"] <= pd.Timestamp.today() + pd.Timedelta(days=7)),
            "valor",
        ].sum()
    ) if not df.empty else 0.0

    c1.metric("A pagar (período)", f"R$ {total_a_pagar:,.2f}")
    c2.metric("Pago (período)", f"R$ {total_pago:,.2f}")
    c3.metric("Qtd. boletos", f"{qtd_boletos}")
    c4.metric("A vencer em 7 dias", f"R$ {prox_7:,.2f}")

    st.divider()

    # --- Matriz (Empresa x Data) ---
    st.markdown("### 📊 Matriz Empresa × Data")
    if df.empty:
        st.warning("Sem dados para o filtro selecionado.")
    else:
        usar_data = st.radio("Agrupar por:", ["Data de vencimento", "Data de pagamento"], horizontal=True)
        col_date = "data_vencimento" if usar_data == "Data de vencimento" else "data_pagamento"
        df_plot = df.copy()
        if col_date == "data_pagamento":
            df_plot = df_plot[df_plot["status"] == "PAGO"]  # somente pagos têm data_pagamento
        df_plot = df_plot.dropna(subset=[col_date])

        # garante coluna somente com data (sem hora)
        df_plot["dia"] = df_plot[col_date].dt.date
        pivot = pd.pivot_table(
            df_plot,
            index="empresa_pagadora",
            columns="dia",
            values="valor",
            aggfunc="sum",
            fill_value=0.0,
        )
        pivot = pivot.sort_index()
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        st.dataframe(
            pivot,
            use_container_width=True,
        )

    st.divider()

    # --- Editor de dados (inline) ---
    st.markdown("### ✏️ Edição Rápida (status, datas, etc.)")
    df_edit = df.copy()
    if not df_edit.empty:
        df_edit = df_edit.sort_values(["data_vencimento", "empresa_pagadora", "beneficiario"])  # ordenação útil
        cfg = {
            "id": st.column_config.NumberColumn("ID"),
            # usa SelectboxColumn se todos os valores estão na lista EMPRESAS; senão, deixa TextColumn para não travar
            "empresa_pagadora": (
                st.column_config.SelectboxColumn("Empresa", options=EMPRESAS)
                if set(df_edit["empresa_pagadora"].dropna().unique()).issubset(set(EMPRESAS)) and len(EMPRESAS) > 0
                else st.column_config.TextColumn("Empresa")
            ),
            "beneficiario": st.column_config.TextColumn("Beneficiário"),
            "banco": st.column_config.TextColumn("Banco"),
            "setor": st.column_config.TextColumn("Setor"),
            "valor": BRL,
            "status": st.column_config.SelectboxColumn("Status", options=["A_PAGAR", "PAGO", "CANCELADO"]),
            "data_registro": DATE_COL,
            "data_vencimento": DATE_COL,
            "data_pagamento": DATE_COL,
            "arquivo_nome": st.column_config.TextColumn("Arquivo (nome)", help="Nome do PDF anexado, se houver"),
            "obs": st.column_config.TextColumn("Observações"),
        }
        edited = st.data_editor(
            df_edit,
            column_config=cfg,
            num_rows="fixed",
            use_container_width=True,
            key="editor1",
        )
        # Identifica linhas alteradas
        changed_rows = [i for i in range(len(df_edit)) if not edited.iloc[i].equals(df_edit.iloc[i])]
        colb1, colb2, colb3 = st.columns([1, 1, 6])
        with colb1:
            if st.button("💾 Salvar alterações", disabled=(len(changed_rows) == 0)):
                update_rows(edited, changed_rows)
                st.success("Alterações salvas.")
                force_rerun()
        with colb2:
            ids_to_delete = st.multiselect("Excluir por ID", edited["id"].astype(int).tolist())
            if st.button("🗑️ Excluir selecionados", disabled=(len(ids_to_delete) == 0)):
                delete_by_ids([int(x) for x in ids_to_delete])
                st.warning("Registros excluídos.")
                force_rerun()

    # --- Importador CSV ---
    st.divider()
    st.markdown("### ⬆️ Importar CSV de boletos")
    st.caption(
        "Esperado: colunas `DATA;EMPRESA PAGADORA;VALOR BOLETO;DATA VENCIMENTO;BENEFICIÁRIO;BANCO;SETOR`.\n"
        "Datas no formato DD/MM/AAAA. Cabeçalho é ignorado quanto a acentos/maiúsculas."
    )
    file = st.file_uploader("Selecione um CSV", type=["csv"])
    if file is not None:
        raw = file.read().decode("utf-8", errors="ignore")
        try:
            df_csv = pd.read_csv(StringIO(raw), sep=";")
        except Exception:
            df_csv = pd.read_csv(StringIO(raw), sep=",")

        cols = {c.strip().lower(): c for c in df_csv.columns}

        def find_col(*aliases):
            for a in aliases:
                if a in cols:
                    return cols[a]
            return None

        c_data = find_col("data", "data registro", "data_registro")
        c_emp = find_col("empresa pagadora", "empresa", "empresa_pagadora")
        c_val = find_col("valor boleto", "valor", "valor_boleto")
        c_vct = find_col("data vencimento", "vencimento", "data_vencimento")
        c_ben = find_col("beneficiário", "beneficiario")
        c_bco = find_col("banco")
        c_set = find_col("setor")

        required = [c_data, c_emp, c_val, c_vct, c_ben]
        if any(c is None for c in required):
            st.error("CSV sem colunas mínimas. Verifique cabeçalhos.")
        else:
            def to_iso(d):
                if pd.isna(d):
                    return None
                try:
                    return datetime.strptime(str(d).strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
                except Exception:
                    try:
                        return pd.to_datetime(d, dayfirst=True, errors="coerce").strftime("%Y-%m-%d")
                    except Exception:
                        return None

            inserted = 0
            for _, r in df_csv.iterrows():
                # padroniza empresa: tenta casar com EMPRESAS (case-insensitive)
                raw_emp = str(r[c_emp]).strip()
                emp_pad = next((e for e in EMPRESAS if e.lower() == raw_emp.lower()), raw_emp)

                payload = {
                    "data_registro": to_iso(r[c_data]),
                    "empresa_pagadora": emp_pad,
                    "valor": float(str(r[c_val]).replace(".", "").replace(",", ".")),
                    "data_vencimento": to_iso(r[c_vct]),
                    "beneficiario": str(r[c_ben]).strip(),
                    "banco": str(r[c_bco]).strip() if c_bco else "",
                    "setor": str(r[c_set]).strip() if c_set else "",
                    "status": "A_PAGAR",
                    "data_pagamento": None,
                    "obs": "",
                    "arquivo_bytes": None,
                    "arquivo_nome": None,
                }
                if not payload["data_registro"]:
                    payload["data_registro"] = date.today().strftime("%Y-%m-%d")
                if not payload["empresa_pagadora"] or not payload["data_vencimento"]:
                    continue
                insert_boleto(payload)
                inserted += 1
            st.success(f"Importação concluída. Registros inseridos: {inserted}.")
            force_rerun()

    # --- Exportar CSV ---
    if not df.empty:
        st.download_button(
            "📥 Baixar CSV (dados filtrados)",
            data=df.to_csv(index=False, sep=";", decimal=","),
            file_name=f"contas_a_pagar_{date.today().isoformat()}.csv",
            mime="text/csv",
        )

    # --- Downloads de PDFs anexados ---
    st.divider()
    st.markdown("### 📎 Boletos anexados (PDF)")
    if df.empty:
        st.caption("Sem anexos para os filtros atuais.")
    else:
        # Busca apenas IDs do df filtrado e gera botões
        anexos_df = df[["id", "beneficiario", "empresa_pagadora", "data_vencimento", "valor"]].copy()
        if anexos_df.empty:
            st.caption("Sem anexos para os filtros atuais.")
        else:
            for _, r in anexos_df.iterrows():
                pdf_bytes, pdf_name = fetch_pdf(int(r["id"]))
                if not pdf_bytes:
                    continue
                colA, colB, colC, colD = st.columns([4, 3, 3, 2])
                colA.write(f"**ID {int(r['id'])}** — {r['beneficiario']} — {r['empresa_pagadora']}")
                colB.write(f"Venc.: {r['data_vencimento'].date() if pd.notna(r['data_vencimento']) else '-'}")
                colC.write(f"Valor: R$ {float(r['valor']):,.2f}")
                colD.download_button(
                    "⬇️ Baixar PDF",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    key=f"dl_{int(r['id'])}",
                )

# Fim do app_pg.py
