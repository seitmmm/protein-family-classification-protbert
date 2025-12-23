import os
import sys
import numpy as np
import torch
import streamlit as st
from io import StringIO
from Bio import SeqIO
import py3Dmol
import tempfile  # сейчас не используем, но пусть будет

# ---- поправляем sys.path, чтобы видеть utils и models ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.embedder import ProtBertEmbedder
from utils.esmfold_wrapper import ESMFoldWrapper
from models.inference_ensemble import load_ensemble
from mutations_web import mutation_stress_test  # <--- НОВОЕ


# ========= кэшируем базовые ресурсы =========

@st.cache_resource
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_embedder():
    device = get_device()
    return ProtBertEmbedder(device=device)


@st.cache_resource
def get_ensemble():
    return load_ensemble()


@st.cache_resource
def get_esmfold():
    """
    Лениво загружаем ESMFold.
    """
    device = get_device()
    if device == "cuda":
        return ESMFoldWrapper(device="cuda")
    else:
        return ESMFoldWrapper(device="cpu")


# ========= вспомогательные функции =========

def parse_sequence_from_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    if text.startswith(">"):
        handle = StringIO(text)
        record = SeqIO.read(handle, "fasta")
        return str(record.seq)
    else:
        seq = "".join(text.split())
        return seq.upper()


def read_fasta_file(uploaded_file) -> str:
    content = uploaded_file.read().decode()
    handle = StringIO(content)
    record = SeqIO.read(handle, "fasta")
    return str(record.seq)


def get_base_model_probs(embedding: np.ndarray, ensemble: dict):
    """
    RF, MLP, KAN probabilities for one embedding.
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    device = ensemble["device"]
    rf = ensemble["rf"]
    mlp = ensemble["mlp"]
    kan = ensemble["kan"]
    scaler_mlp = ensemble["scaler_mlp"]
    scaler_kan = ensemble["scaler_kan"]
    num_classes = ensemble["num_classes"]

    probs_rf = rf.predict_proba(embedding)[0]

    X_mlp = scaler_mlp.transform(embedding)
    X_mlp_t = torch.from_numpy(X_mlp).float().to(device)
    with torch.no_grad():
        logits_mlp = mlp(X_mlp_t)
        probs_mlp = torch.softmax(logits_mlp, dim=1).cpu().numpy()[0]

    X_kan = scaler_kan.transform(embedding)
    X_kan_t = torch.from_numpy(X_kan).float().to(device)
    with torch.no_grad():
        logits_kan = kan(X_kan_t)
        probs_kan = torch.softmax(logits_kan, dim=1).cpu().numpy()[0]

    return probs_rf, probs_mlp, probs_kan


def predict_with_mode(embedding: np.ndarray,
                      ensemble: dict,
                      mode: str = "Meta-ensemble",
                      topk: int = 3,
                      threshold: float = 0.55):
    """
    mode:
      - "RF"
      - "MLP v2"
      - "KAN"
      - "Soft ensemble"
      - "Meta-ensemble"
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    family_to_id = ensemble["family_to_id"]
    id_to_family = ensemble["id_to_family"]
    num_classes = ensemble["num_classes"]

    probs_rf, probs_mlp, probs_kan = get_base_model_probs(embedding, ensemble)

    if mode == "RF":
        probs = probs_rf
    elif mode == "MLP v2":
        probs = probs_mlp
    elif mode == "KAN":
        probs = probs_kan
    elif mode == "Soft ensemble":
        w_rf, w_mlp, w_kan = 0.2, 0.4, 0.4
        probs = w_rf * probs_rf + w_mlp * probs_mlp + w_kan * probs_kan
    elif mode == "Meta-ensemble":
        meta = ensemble["meta"]
        X_meta = np.concatenate(
            [probs_rf.reshape(1, -1),
             probs_mlp.reshape(1, -1),
             probs_kan.reshape(1, -1)],
            axis=1
        )  # (1, 24)
        probs_meta = meta.predict_proba(X_meta)[0]
        classes = meta.classes_
        full_probs = np.zeros(num_classes, dtype=float)
        full_probs[classes] = probs_meta
        probs = full_probs
    else:
        raise ValueError(f"Unknown mode: {mode}")

    idx_sorted = np.argsort(probs)[::-1]
    topk_idx = idx_sorted[:topk]
    topk_list = [(id_to_family[i], float(probs[i])) for i in topk_idx]

    best_idx = topk_idx[0]
    best_prob = float(probs[best_idx])

    if best_prob < threshold:
        predicted_family = "unknown"
    else:
        predicted_family = id_to_family[best_idx]

    raw_probs = {id_to_family[i]: float(probs[i]) for i in range(num_classes)}

    return {
        "predicted_family": predicted_family,
        "max_prob": best_prob,
        "topk": topk_list,
        "raw_probs": raw_probs,
    }


def show_pdb_3d_from_str(pdb_str: str):
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    html = view._make_html()
    st.components.v1.html(html, height=600)


def show_pdb_3d_from_file(pdb_file):
    pdb_bytes = pdb_file.read()
    pdb_str = pdb_bytes.decode() if isinstance(pdb_bytes, (bytes, bytearray)) else pdb_bytes
    show_pdb_3d_from_str(pdb_str)


def run_single_prediction(seq, model_choice, threshold, topk):
    """
    Общая функция: считает embedding, делает предсказание
    и возвращает emb, result, embedder, ensemble.
    """
    embedder = get_embedder()
    ensemble = get_ensemble()
    emb = embedder.get_embedding(seq)
    result = predict_with_mode(
        emb,
        ensemble,
        mode=model_choice,
        topk=topk,
        threshold=threshold
    )
    return emb, result, embedder, ensemble


# ========= Streamlit UI =========

st.set_page_config(page_title="Protein Family Prediction", layout="wide")

st.title("Protein Family Prediction with ProtBERT + RF/MLP/KAN/Ensemble")
st.markdown(
    """
    Enter the amino acid sequence or upload a FASTA file,
select a model, and get a protein family prediction.
If needed, you can generate a 3D structure using ESMFold.
    """
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Input sequence")

    uploaded_fasta = st.file_uploader("Upload FASTA file", type=["fasta", "fa"])
    text_input = st.text_area("Or paste sequence / FASTA here", height=200)

    if uploaded_fasta is not None:
        seq = read_fasta_file(uploaded_fasta)
        st.write(f"Loaded sequence length: **{len(seq)}** amino acids.")
    elif text_input.strip():
        seq = parse_sequence_from_text(text_input)
        if seq:
            st.write(f"Parsed sequence length: **{len(seq)}** amino acids.")
        else:
            seq = ""
            st.warning("Could not parse sequence.")
    else:
        seq = ""

with col_right:
    st.subheader("Model & Settings")

    model_choice = st.selectbox(
        "Model",
        ["Meta-ensemble", "Soft ensemble", "RF", "MLP v2", "KAN"],
        index=0
    )

    threshold = st.slider(
        "Unknown threshold (no max probability)",
        min_value=0.3,
        max_value=0.9,
        value=0.55,
        step=0.01
    )

    topk = st.slider("Top-k predictions", min_value=1, max_value=8, value=3, step=1)

    generate_3d = st.checkbox(
        "Generate 3D structure from sequence with ESMFold (experimental, can be slow)",
        value=False
    )

    # Кнопка только для базового предсказания
    run_pred_button = st.button("Predict family")

    st.markdown("---")
    st.subheader("Mutation robustness test")

    mutation_mode = st.radio(
        "Mutation mode",
        ["Absolute (# mutations)", "Percentage of sequence"],
        index=0,
    )

    if mutation_mode == "Absolute (# mutations)":
        k_abs = st.slider(
            "Number of mutated positions (k)",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )
        k_percent = None
    else:
        k_percent = st.slider(
            "Mutation percent of sequence",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Процент от длины последовательности, который будет случайно мутирован."
        )
        k_abs = None

    n_mutants = st.slider(
        "Number of mutants to sample",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Чем больше мутантов, тем точнее оценка, но тем дольше считается."
    )

    # Отдельная кнопка для мутаций
    run_mut_button = st.button("Run mutation test")

    st.markdown("---")
    st.subheader("3D Structure Viewer (optional PDB)")
    pdb_file = st.file_uploader("Upload PDB file", type=["pdb"])

    if pdb_file is not None:
        show_pdb_3d_from_file(pdb_file)


# ====== БАЗОВОЕ ПРЕДСКАЗАНИЕ ======

if run_pred_button:
    if not seq:
        st.error("Please provide a sequence or FASTA file.")
    else:
        if generate_3d and len(seq) > 600:
            st.warning(
                f"Sequence length = {len(seq)} aa, too long for ESMFold on 6GB VRAM.\n"
                f"For 3D demonstration, it is better to use proteins up to ~400–600 aa."
            )

        with st.spinner("Computing embedding and running model..."):
            emb, result, embedder, ensemble = run_single_prediction(
                seq, model_choice, threshold, topk
            )

        # сохраняем в session_state, чтобы переиспользовать в мутациях
        st.session_state["last_seq"] = seq
        st.session_state["last_emb"] = emb
        st.session_state["last_result"] = result

        st.success("Prediction complete.")
        st.write(f"**Selected model:** {model_choice}")
        st.write(f"**Predicted family:** `{result['predicted_family']}`")
        st.write(f"**Max probability:** {result['max_prob']:.3f}")

        st.write("**Top predictions:**")
        for fam, prob in result["topk"]:
            st.write(f"- {fam}: {prob:.3f}")

        with st.expander("Show all class probabilities"):
            import pandas as pd
            probs_df = pd.DataFrame(
                {
                    "Family": list(result["raw_probs"].keys()),
                    "Probability": list(result["raw_probs"].values())
                }
            ).sort_values("Probability", ascending=False)
            st.dataframe(probs_df, width="stretch")

        # --- ESMFold 3D ---
        if generate_3d:
            st.markdown("---")
            st.subheader("Predicted 3D structure (ESMFold)")

            try:
                esmfold = get_esmfold()
                with st.spinner("Running ESMFold (this may take some time)..."):
                    pdb_str = esmfold.predict_pdb(seq)
                show_pdb_3d_from_str(pdb_str)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    st.warning("CUDA out of memory on GPU. Retrying ESMFold on CPU (slower)...")
                    try:
                        esmfold_cpu = ESMFoldWrapper(device="cpu")
                        with st.spinner("Running ESMFold on CPU..."):
                            pdb_str = esmfold_cpu.predict_pdb(seq)
                        show_pdb_3d_from_str(pdb_str)
                    except Exception as e2:
                        st.error(f"ESMFold CPU fallback failed: {e2}")
                        st.info("You can still upload a PDB file above to see 3D structure.")
                else:
                    st.error(f"ESMFold failed: {e}")
                    st.info("You can still upload a PDB file above to see 3D structure.")
            except Exception as e:
                st.error(f"ESMFold failed: {e}")
                st.info("You can still upload a PDB file above to see 3D structure.")


# ====== МУТАЦИОННЫЙ СТРЕСС-ТЕСТ ======

if run_mut_button:
    if not seq:
        st.error("Please provide a sequence or FASTA file.")
    else:
        # пробуем переиспользовать последнее предсказание, если последовательность та же
        last_seq = st.session_state.get("last_seq")
        if last_seq == seq and "last_emb" in st.session_state and "last_result" in st.session_state:
            emb = st.session_state["last_emb"]
            base_result = st.session_state["last_result"]
            embedder = get_embedder()
            ensemble = get_ensemble()
        else:
            # если до этого не предсказывали или seq поменялась — считаем заново
            with st.spinner("Computing embedding and base prediction for mutation test..."):
                emb, base_result, embedder, ensemble = run_single_prediction(
                    seq, model_choice, threshold, topk
                )

        base_family = base_result["predicted_family"]
        if base_family == "unknown":
            st.info(
                "Base prediction is 'unknown', mutation test is not very informative.\n"
                "At first, it is better for the model to confidently assign a protein to a certain family."
            )
        else:
            if mutation_mode == "Absolute (# mutations)":
                k_mut = min(k_abs, len(seq))
                k_text = f"{k_mut} positions"
            else:
                k_mut = max(1, int(round(len(seq) * k_percent / 100)))
                k_text = f"{k_percent}% of positions (≈ {k_mut} aa)"

            # predict_fn, который принимает embedding и зовёт predict_with_mode
            def predict_fn(embedding_np: np.ndarray):
                return predict_with_mode(
                    embedding_np,
                    ensemble,
                    mode=model_choice,
                    topk=topk,
                    threshold=threshold,
                )

            with st.spinner(f"Running mutation test ({k_text}, {n_mutants} mutants)..."):
                mut_result = mutation_stress_test(
                    seq=seq,
                    k=k_mut,
                    n_mutants=n_mutants,
                    base_family=base_family,
                    embedder=embedder,
                    predict_fn=predict_fn,
                )

            st.markdown("---")
            st.subheader("Mutation robustness result")

            st.write(f"**Base predicted family:** `{base_family}`")
            st.write(f"**Mutation setting:** {k_text}")
            st.write(f"**Number of mutants:** {mut_result['n_mutants']}")

            st.write(
                f"- Correct predictions: "
                f"**{mut_result['n_correct']}/{mut_result['n_mutants']}** "
                f"(accuracy = {mut_result['accuracy'] * 100:.1f}%)"
            )
            st.write(
                f"- Mean probability of base family over mutants: "
                f"**{mut_result['mean_prob']:.3f}**"
            )

            st.caption(
                "The slower these values fall as the number/percentage of mutations increases, "
                "the more robust the model is to random changes in the sequence."
            )
