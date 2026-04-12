from acualfiletorun import *

import gradio as gr


# --- Gradio UI with Tabs ---
with gr.Blocks() as pdf_qa_ui:
    gr.Markdown("# 📄 PDF-based RAG Q&A")
    gr.Markdown("Powered by **Pinecone + HuggingFace + Gemini**")

    with gr.Tabs():

        # ======= TAB 1: Upload PDF =======
        with gr.Tab("📂 Upload PDF"):
            gr.Markdown("### Upload your PDF to index it into the Vector Database")
            file_input = gr.File(label="Select PDF File")
            upload_btn = gr.Button("Upload & Index", variant="primary")
            upload_output = gr.Textbox(label="Upload Status")

            upload_btn.click(
                fn=upload_tab,
                inputs=file_input,
                outputs=upload_output
            )

        # ======= TAB 2: Ask Question =======
        with gr.Tab("❓ Ask a Question"):
            gr.Markdown("### Ask a question based on the indexed PDF")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is monitoring in distributed system?"
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(label="💬 Answer")

            ask_btn.click(
                fn=qa_tab,
                inputs=question_input,
                outputs=answer_output
            )

pdf_qa_ui.launch(theme="soft")  # ✅ theme in launch()
