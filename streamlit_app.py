import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, configs
from src.questions_processing import QuestionsProcessor
import json

# 设置页面标题
st.title("RAG Challenge Q&A System")

# 加载 Pipeline 配置
config_options = list(configs.keys())
selected_config = st.selectbox("Select a configuration", config_options, index=config_options.index("base"))

# 初始化 Pipeline
@st.cache_resource
def load_pipeline(config_name):
    root_path = Path.cwd() / "data" / "test_set"
    run_config = configs[config_name]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    # 我们只需要 QuestionsProcessor，所以直接实例化它
    processor = QuestionsProcessor(
        vector_db_dir=pipeline.paths.vector_db_dir,
        documents_dir=pipeline.paths.documents_dir,
        questions_file_path=None,  # 我们将通过 UI 接收问题
        new_challenge_pipeline=True,
        subset_path=pipeline.paths.subset_path,
        parent_document_retrieval=run_config.parent_document_retrieval,
        llm_reranking=run_config.llm_reranking,
        llm_reranking_sample_size=run_config.llm_reranking_sample_size,
        top_n_retrieval=run_config.top_n_retrieval,
        parallel_requests=run_config.parallel_requests,
        api_provider=run_config.api_provider,
        answering_model=run_config.answering_model,
        full_context=run_config.full_context
    )
    return processor

processor = load_pipeline(selected_config)

# 用户输入
question = st.text_area("Enter your question here:")
schema_options = ["text", "number", "comparative"] # 假设这些是可能的类型
schema = st.selectbox("Select the question type (schema)", schema_options)


if st.button("Get Answer"):
    if question and schema:
        with st.spinner("Finding the answer..."):
            try:
                # 调用处理问题的核心逻辑
                answer_dict = processor.process_question(question, schema)
                
                st.subheader("Answer")
                st.write(answer_dict.get("final_answer", "N/A"))
                
                st.subheader("Thinking Process")
                st.text(answer_dict.get("step_by_step_analysis", "No analysis available."))
                
                st.subheader("Reasoning Summary")
                st.text(answer_dict.get("reasoning_summary", "No summary available."))

                st.subheader("References")
                references = answer_dict.get("references", [])
                if references:
                    for ref in references:
                        st.write(f"- PDF: {ref['pdf_sha1']}, Page: {ref['page_index']}")
                else:
                    st.write("No references found.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question and select a schema.")
