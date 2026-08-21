import streamlit as st

from utils import extract_pdf, create_vector_text

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(
    page_title="Help4Code Resume Analyzer",
    page_icon=""
)

st.title("Help4Code Resume Analyzer AI")


resume_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

jd_text = st.text_area(
    "Paste Job Description"
)


if st.button("Analyze"):

    if resume_file is None:
        st.warning("Please upload your resume.")
        st.stop()

    if not jd_text.strip():
        st.warning("Please enter the job description.")
        st.stop()

    with st.spinner("Analyzing your resume..."):

        # Extract resume
        resume_text = extract_pdf(resume_file)

        if not resume_text.strip():
            st.error("Could not extract text from the PDF.")
            st.stop()

        # Combine resume and JD
        combined_text = (
            "RESUME:\n"
            + resume_text
            + "\n\n"
            + "JOB DESCRIPTION:\n"
            + jd_text
        )

        # Create vector database
        vectorstore = create_vector_text(combined_text)

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # Ollama LLM
        llm = Ollama(
            model="gemma2:2b"
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are an AI placement coach for Help4Code.

Analyze the candidate's resume against the job description.

Context:
{context}

Question:
{question}

Provide the following:

1. Skills Gap Analysis
2. Missing Technologies
3. ATS Score from 0-100
4. 10 Technical Interview Questions
5. Resume Improvement Suggestions

Give practical and specific recommendations.
"""
        )

        # RAG chain
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Run chain
        response = chain.invoke(
            "Analyze the resume against the job description."
        )

    st.subheader("Analysis Result")
    st.write(response)
